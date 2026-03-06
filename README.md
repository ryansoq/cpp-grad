# cpp-grad 🛡️

一個微型自動微分引擎，用現代 C++17 從零實現。**記憶體安全**。

> 零裸指標、零 new/delete、ASan 全程護航。Header-only，~200 行。

---

## 安裝與使用

### 前置需求

```bash
# 需要支援 C++17 的編譯器
g++ --version   # GCC 7+ 或 Clang 5+
```

### 方法一：Clone 下來跑測試

```bash
git clone https://github.com/ryansoq/cpp-grad.git
cd cpp-grad

# 跑測試（27 個，含 AddressSanitizer 記憶體檢查）
make test

# 跑範例
make basic && ./basic
```

### 方法二：作為 Header-only Library 引入

把 `include/cpp_grad.hpp` 複製到你的專案：

```cpp
#include "cpp_grad.hpp"

int main() {
    auto a = Value::create(-4.0);
    auto b = Value::create(2.0);

    // 前向：operator overloading 自動建圖
    auto c = a + b;
    auto d = (a * b) + b->pow_val(3.0);
    auto e = c - d;
    auto f = e->pow_val(2.0);

    // 反向：一行算出所有梯度
    f->backward();

    std::cout << "∂f/∂a = " << a->grad << std::endl;
    std::cout << "∂f/∂b = " << b->grad << std::endl;
}
```

### C++ 新手提示

- `Value::create(3.0)` — 回傳 `shared_ptr<Value>`，自動管理記憶體
- `a + b` — operator overloading，直接用 `+` `*` `-` `/`
- `->backward()` — 反向傳播，算完後讀 `->grad`
- 不需要 `delete`，`shared_ptr` 離開 scope 自動釋放

---

## 支援的運算

| 運算 | 用法 | forward | backward (∂/∂x) |
|------|------|---------|------------------|
| 加法 | `a + b` | x + y | 1 |
| 減法 | `a - b` | x - y | 1, -1 |
| 乘法 | `a * b` | x × y | y, x |
| 除法 | `a / b` | x / y | 1/y, -x/y² |
| 負號 | `a->neg()` | -x | -1 |
| 次方 | `a->pow_val(n)` | xⁿ | n·xⁿ⁻¹ |
| 指數 | `a->exp_val()` | eˣ | eˣ |
| 對數 | `a->ln_val()` | ln(x) | 1/x |
| ReLU | `a->relu()` | max(0,x) | x>0 ? 1 : 0 |
| tanh | `a->tanh_val()` | tanh(x) | 1 - tanh²(x) |
| sigmoid | `a->sigmoid()` | σ(x) | σ(x)(1-σ(x)) |

---

## 記憶體安全設計

### 傳統 C++ 的問題

```cpp
// ❌ 裸指標 — 忘了 delete 就 memory leak
Value* v = new Value(3.0);
// ... 忘了 delete v;

// ❌ Double free
delete v;
delete v;  // 💥 crash
```

### cpp-grad 的解法

```cpp
// ✅ shared_ptr — 自動引用計數，沒人用就自動釋放
auto v = Value::create(3.0);
// 不需要 delete，離開 scope 自動回收

// ✅ weak_ptr — 閉包引用自己時避免循環引用
out->backward_fn_ = [self, w = std::weak_ptr<Value>(out)] {
    if (auto out = w.lock()) { ... }
};
```

| 安全機制 | 解決什麼 |
|---------|---------|
| `shared_ptr` | 自動引用計數，不可能 leak |
| `weak_ptr` | 避免閉包造成的循環引用 |
| `Value::create()` 工廠 | 禁止裸 `new` |
| `-fsanitize=address` | 編譯時加 ASan，跑的時候抓任何記憶體錯誤 |
| `-fsanitize=undefined` | 抓未定義行為（溢位、null deref 等）|

---

## 測試

```bash
make test
```

27 個測試覆蓋：
- 每個 op 的 forward + backward
- 鏈式法則、共用節點梯度累加
- micrograd 官方範例數值驗證（精確到小數後 4 位）
- ASan 全程開啟 — 記憶體問題零容忍

---

## 設計哲學

- **記憶體安全**：C++ 也能寫出不 leak 的代碼
- **Header-only**：一個 `.hpp` 搞定，不需要編譯 library
- **最小化**：只做自動微分，不做 tensor、不做 nn layer
- **可讀性**：每個 op 的梯度公式寫在註解裡
- **可擴展**：要組 MLP、CNN？include cpp_grad 自己蓋

---

## 與 rust-grad 的對照

| | [rust-grad](https://github.com/ryansoq/rust-grad) | cpp-grad |
|--|-----------|----------|
| 語言 | Rust | C++17 |
| 記憶體安全 | 編譯器保證 (ownership) | shared_ptr + ASan |
| API 風格 | `&a + &b` | `a + b` (shared_ptr) |
| 建置 | `cargo test` | `make test` |
| 測試數 | 28 | 27 |
| micrograd 驗證 | ✅ | ✅ |

同一個 autograd engine，兩種語言實現。數學相同，安全機制不同。

---

## License

MIT
