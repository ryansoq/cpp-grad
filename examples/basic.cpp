// cpp-grad 基本範例
#include "cpp_grad.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== cpp-grad 基本範例 ===\n" << std::endl;

    // 1. 基本四則運算
    std::cout << "--- 1. 基本四則運算 ---" << std::endl;
    auto a = Value::create(2.0);
    auto b = Value::create(3.0);
    std::cout << "a=" << a->data << ", b=" << b->data << std::endl;
    std::cout << "a+b = " << (a + b)->data << std::endl;
    std::cout << "a*b = " << (a * b)->data << std::endl;
    std::cout << "a-b = " << (a - b)->data << std::endl;
    std::cout << "a/b = " << (a / b)->data << std::endl;

    // 2. 反向傳播
    std::cout << "\n--- 2. 反向傳播（梯度計算）---" << std::endl;
    auto x = Value::create(3.0);
    auto y = Value::create(4.0);
    auto z = x * y;
    z->backward();
    std::cout << "z = x * y = " << x->data << " * " << y->data << " = " << z->data << std::endl;
    std::cout << "∂z/∂x = " << x->grad << " (應該 = y = 4)" << std::endl;
    std::cout << "∂z/∂y = " << y->grad << " (應該 = x = 3)" << std::endl;

    // 3. 鏈式法則
    std::cout << "\n--- 3. 鏈式法則 ---" << std::endl;
    a = Value::create(2.0);
    b = Value::create(3.0);
    auto c = a * b;
    auto d = c->pow_val(2.0);
    d->backward();
    std::cout << "d = (a*b)² = (" << a->data << "*" << b->data << ")² = " << d->data << std::endl;
    std::cout << "∂d/∂a = " << a->grad << " (應該 = 2ab² = 36)" << std::endl;
    std::cout << "∂d/∂b = " << b->grad << " (應該 = 2a²b = 24)" << std::endl;

    // 4. 激活函數
    std::cout << "\n--- 4. 激活函數 ---" << std::endl;
    x = Value::create(-2.0);
    std::cout << "x = " << x->data << std::endl;
    std::cout << "relu(x) = " << x->relu()->data << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "sigmoid(x) = " << x->sigmoid()->data << std::endl;
    std::cout << "tanh(x) = " << x->tanh_val()->data << std::endl;

    x = Value::create(2.0);
    std::cout << "\nx = " << x->data << std::endl;
    std::cout << "relu(x) = " << x->relu()->data << std::endl;
    std::cout << "sigmoid(x) = " << x->sigmoid()->data << std::endl;
    std::cout << "tanh(x) = " << x->tanh_val()->data << std::endl;

    // 5. micrograd 官方範例
    std::cout << "\n--- 5. micrograd 官方範例驗證 ---" << std::endl;
    a = Value::create(-4.0);
    b = Value::create(2.0);
    c = a + b;
    d = (a * b) + b->pow_val(3.0);
    c = c + (c + Value::create(1.0));
    c = c + (Value::create(1.0) + c + a->neg());
    auto d2 = d * Value::create(2.0);
    auto ba_relu = (b + a)->relu();
    d = d + (d2 + ba_relu);
    auto d3 = Value::create(3.0) * d;
    auto bna_relu = (b + a->neg())->relu();
    d = d + (d3 + bna_relu);
    auto e = c - d;
    auto f = e->pow_val(2.0);
    auto g = f / Value::create(2.0);
    g = g + (Value::create(10.0) / f);

    std::cout << "g = " << g->data << " (micrograd: 24.7041)" << std::endl;
    g->backward();
    std::cout << "∂g/∂a = " << a->grad << " (micrograd: 138.8338)" << std::endl;
    std::cout << "∂g/∂b = " << b->grad << " (micrograd: 645.5773)" << std::endl;

    std::cout << "\n✅ 完成！" << std::endl;
    return 0;
}
