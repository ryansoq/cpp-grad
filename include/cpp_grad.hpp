// cpp-grad: 一個微型自動微分引擎（現代 C++17，記憶體安全）
// 零裸指標、零 new/delete，全部用 shared_ptr 管理
//
// 用法：
//   auto a = Value::create(-4.0);
//   auto b = Value::create(2.0);
//   auto c = a + b;
//   c->backward();
//   std::cout << a->grad << std::endl;

#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

class Value : public std::enable_shared_from_this<Value> {
public:
    using Ptr = std::shared_ptr<Value>;

    double data;    // 前向值
    double grad;    // 反向梯度 ∂L/∂self

private:
    std::vector<Ptr> prev_;              // 上游節點（計算圖的邊）
    std::function<void()> backward_fn_;  // 反向傳播閉包
    std::string op_;                     // 運算標籤（debug 用）

    // 私有建構子 — 強制使用 create() 工廠函數
    explicit Value(double data) : data(data), grad(0.0) {}

public:
    // ===== 工廠函數 =====

    /// 建立一個新的 Value 節點
    static Ptr create(double data) {
        // 不能用 make_shared（建構子是 private），用 shared_ptr + new
        // 這是唯一允許 new 的地方，被 shared_ptr 包住所以安全
        return Ptr(new Value(data));
    }

    // ===== 前向運算 =====

    /// 加法: z = x + y
    /// ∂z/∂x = 1, ∂z/∂y = 1
    friend Ptr operator+(const Ptr& a, const Ptr& b) {
        auto out = create(a->data + b->data);
        out->prev_ = {a, b};
        out->op_ = "+";
        out->backward_fn_ = [a, b, w = std::weak_ptr<Value>(out)] {
            if (auto out = w.lock()) {
                a->grad += out->grad;       // ∂z/∂x = 1
                b->grad += out->grad;       // ∂z/∂y = 1
            }
        };
        return out;
    }

    /// 乘法: z = x * y
    /// ∂z/∂x = y, ∂z/∂y = x
    friend Ptr operator*(const Ptr& a, const Ptr& b) {
        auto out = create(a->data * b->data);
        out->prev_ = {a, b};
        out->op_ = "*";
        double a_val = a->data, b_val = b->data;
        out->backward_fn_ = [a, b, a_val, b_val, w = std::weak_ptr<Value>(out)] {
            if (auto out = w.lock()) {
                a->grad += b_val * out->grad;   // ∂(xy)/∂x = y
                b->grad += a_val * out->grad;   // ∂(xy)/∂y = x
            }
        };
        return out;
    }

    /// 減法: z = x - y（實作為 x + (-y)）
    friend Ptr operator-(const Ptr& a, const Ptr& b) {
        return a + b->neg();
    }

    /// 除法: z = x / y（實作為 x * y^(-1)）
    friend Ptr operator/(const Ptr& a, const Ptr& b) {
        return a * b->pow_val(-1.0);
    }

    /// 負號: -x
    /// ∂(-x)/∂x = -1
    Ptr neg() {
        auto minus_one = create(-1.0);
        auto self = shared_from_this();
        return self * minus_one;
    }

    /// 次方: x^n
    /// ∂(x^n)/∂x = n * x^(n-1)
    Ptr pow_val(double n) {
        auto self = shared_from_this();
        auto out = create(std::pow(self->data, n));
        out->prev_ = {self};
        out->op_ = "**" + std::to_string(n);
        double self_data = self->data;
        out->backward_fn_ = [self, n, self_data, w = std::weak_ptr<Value>(out)] {
            if (auto out = w.lock()) {
                // ∂(x^n)/∂x = n * x^(n-1)
                self->grad += n * std::pow(self_data, n - 1.0) * out->grad;
            }
        };
        return out;
    }

    /// ReLU: max(0, x)
    /// ∂relu/∂x = x > 0 ? 1 : 0
    Ptr relu() {
        auto self = shared_from_this();
        double out_data = self->data > 0 ? self->data : 0.0;
        auto out = create(out_data);
        out->prev_ = {self};
        out->op_ = "relu";
        out->backward_fn_ = [self, out_data, w = std::weak_ptr<Value>(out)] {
            if (auto out = w.lock()) {
                self->grad += (out_data > 0 ? 1.0 : 0.0) * out->grad;
            }
        };
        return out;
    }

    /// 指數: e^x
    /// ∂(e^x)/∂x = e^x
    Ptr exp_val() {
        auto self = shared_from_this();
        double e = std::exp(self->data);
        auto out = create(e);
        out->prev_ = {self};
        out->op_ = "exp";
        out->backward_fn_ = [self, e, w = std::weak_ptr<Value>(out)] {
            if (auto out = w.lock()) {
                self->grad += e * out->grad;
            }
        };
        return out;
    }

    /// 自然對數: ln(x)
    /// ∂ln(x)/∂x = 1/x
    Ptr ln_val() {
        auto self = shared_from_this();
        auto out = create(std::log(self->data));
        out->prev_ = {self};
        out->op_ = "ln";
        double self_data = self->data;
        out->backward_fn_ = [self, self_data, w = std::weak_ptr<Value>(out)] {
            if (auto out = w.lock()) {
                self->grad += (1.0 / self_data) * out->grad;
            }
        };
        return out;
    }

    /// tanh: tanh(x)
    /// ∂tanh(x)/∂x = 1 - tanh²(x)
    Ptr tanh_val() {
        auto self = shared_from_this();
        double t = std::tanh(self->data);
        auto out = create(t);
        out->prev_ = {self};
        out->op_ = "tanh";
        out->backward_fn_ = [self, t, w = std::weak_ptr<Value>(out)] {
            if (auto out = w.lock()) {
                self->grad += (1.0 - t * t) * out->grad;
            }
        };
        return out;
    }

    /// sigmoid: σ(x) = 1 / (1 + e^(-x))
    /// ∂σ/∂x = σ(x)(1 - σ(x))
    Ptr sigmoid() {
        auto self = shared_from_this();
        double s = 1.0 / (1.0 + std::exp(-self->data));
        auto out = create(s);
        out->prev_ = {self};
        out->op_ = "sigmoid";
        out->backward_fn_ = [self, s, w = std::weak_ptr<Value>(out)] {
            if (auto out = w.lock()) {
                self->grad += s * (1.0 - s) * out->grad;
            }
        };
        return out;
    }

    // ===== 反向傳播 =====

    /// 從這個節點出發，計算所有上游節點的梯度
    /// 1. DFS 拓撲排序
    /// 2. self.grad = 1.0（∂L/∂L = 1）
    /// 3. 逆序走，逐個呼叫 backward_fn（鏈式法則）
    void backward() {
        std::vector<Ptr> topo;
        std::unordered_set<Value*> visited;

        // 拓撲排序（DFS）
        std::function<void(const Ptr&)> build_topo = [&](const Ptr& v) {
            if (visited.count(v.get())) return;
            visited.insert(v.get());
            for (auto& child : v->prev_) {
                build_topo(child);
            }
            topo.push_back(v);
        };

        build_topo(shared_from_this());

        // ∂L/∂L = 1
        this->grad = 1.0;

        // 逆序走，鏈式法則
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_fn_) {
                (*it)->backward_fn_();
            }
        }
    }

    // ===== Display =====

    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
        return os;
    }
};
