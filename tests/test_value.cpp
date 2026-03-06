// cpp-grad 測試 — 每個 op 都有 forward + backward 測試
// 測試框架：純 C++ assert，不依賴外部庫

#include "cpp_grad.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

#define ASSERT_NEAR(a, b, eps) \
    assert(std::abs((a) - (b)) < (eps) && #a " ≈ " #b)

int tests_passed = 0;
int tests_total = 0;

#define TEST(name) \
    void test_##name(); \
    struct Register_##name { \
        Register_##name() { \
            tests_total++; \
            std::cout << "  " #name "... "; \
            test_##name(); \
            tests_passed++; \
            std::cout << "✅" << std::endl; \
        } \
    } register_##name; \
    void test_##name()

// ===== T02: Value 基礎 =====

TEST(value_new) {
    auto v = Value::create(3.0);
    assert(v->data == 3.0);
    assert(v->grad == 0.0);
}

// ===== T04: Add =====

TEST(add_forward) {
    auto a = Value::create(2.0);
    auto b = Value::create(3.0);
    auto c = a + b;
    assert(c->data == 5.0);
}

TEST(add_backward) {
    auto a = Value::create(2.0);
    auto b = Value::create(3.0);
    auto c = a + b;
    c->backward();
    assert(a->grad == 1.0);     // ∂c/∂a = 1
    assert(b->grad == 1.0);     // ∂c/∂b = 1
}

// ===== T05: Mul =====

TEST(mul_forward) {
    auto a = Value::create(3.0);
    auto b = Value::create(4.0);
    auto c = a * b;
    assert(c->data == 12.0);
}

TEST(mul_backward) {
    auto a = Value::create(3.0);
    auto b = Value::create(4.0);
    auto c = a * b;
    c->backward();
    assert(a->grad == 4.0);     // ∂c/∂a = b = 4
    assert(b->grad == 3.0);     // ∂c/∂b = a = 3
}

// ===== T06: Neg =====

TEST(neg_forward) {
    auto a = Value::create(3.0);
    auto b = a->neg();
    assert(b->data == -3.0);
}

// ===== T07: Sub =====

TEST(sub_forward) {
    auto a = Value::create(5.0);
    auto b = Value::create(2.0);
    auto c = a - b;
    assert(c->data == 3.0);
}

TEST(sub_backward) {
    auto a = Value::create(5.0);
    auto b = Value::create(2.0);
    auto c = a - b;
    c->backward();
    assert(a->grad == 1.0);     // ∂(a-b)/∂a = 1
    assert(b->grad == -1.0);    // ∂(a-b)/∂b = -1
}

// ===== T08: Pow =====

TEST(pow_forward) {
    auto a = Value::create(2.0);
    auto b = a->pow_val(3.0);
    assert(b->data == 8.0);
}

TEST(pow_backward) {
    auto a = Value::create(2.0);
    auto b = a->pow_val(3.0);
    b->backward();
    assert(a->grad == 12.0);    // ∂(x³)/∂x = 3x² = 12
}

// ===== T09: ReLU =====

TEST(relu_forward_positive) {
    auto a = Value::create(5.0);
    assert(a->relu()->data == 5.0);
}

TEST(relu_forward_negative) {
    auto a = Value::create(-3.0);
    assert(a->relu()->data == 0.0);
}

TEST(relu_backward_positive) {
    auto a = Value::create(5.0);
    auto b = a->relu();
    b->backward();
    assert(a->grad == 1.0);
}

TEST(relu_backward_negative) {
    auto a = Value::create(-3.0);
    auto b = a->relu();
    b->backward();
    assert(a->grad == 0.0);
}

// ===== T17: Exp =====

TEST(exp_forward) {
    auto a = Value::create(1.0);
    ASSERT_NEAR(a->exp_val()->data, M_E, 1e-6);
}

TEST(exp_backward) {
    auto a = Value::create(2.0);
    auto b = a->exp_val();
    b->backward();
    ASSERT_NEAR(a->grad, std::exp(2.0), 1e-6);
}

// ===== T17: Ln =====

TEST(ln_forward) {
    auto a = Value::create(M_E);
    ASSERT_NEAR(a->ln_val()->data, 1.0, 1e-6);
}

TEST(ln_backward) {
    auto a = Value::create(3.0);
    auto b = a->ln_val();
    b->backward();
    ASSERT_NEAR(a->grad, 1.0 / 3.0, 1e-6);
}

// ===== T18: Div =====

TEST(div_forward) {
    auto a = Value::create(6.0);
    auto b = Value::create(3.0);
    auto c = a / b;
    ASSERT_NEAR(c->data, 2.0, 1e-6);
}

TEST(div_backward) {
    auto a = Value::create(6.0);
    auto b = Value::create(3.0);
    auto c = a / b;
    c->backward();
    ASSERT_NEAR(a->grad, 1.0 / 3.0, 1e-6);      // ∂(a/b)/∂a = 1/b
    ASSERT_NEAR(b->grad, -2.0 / 3.0, 1e-6);      // ∂(a/b)/∂b = -a/b²
}

// ===== T21: Tanh =====

TEST(tanh_forward) {
    auto a = Value::create(0.0);
    ASSERT_NEAR(a->tanh_val()->data, 0.0, 1e-6);
}

TEST(tanh_backward) {
    auto a = Value::create(1.0);
    auto b = a->tanh_val();
    b->backward();
    double t = std::tanh(1.0);
    ASSERT_NEAR(a->grad, 1.0 - t * t, 1e-6);
}

// ===== T21: Sigmoid =====

TEST(sigmoid_forward) {
    auto a = Value::create(0.0);
    ASSERT_NEAR(a->sigmoid()->data, 0.5, 1e-6);
}

TEST(sigmoid_backward) {
    auto a = Value::create(1.0);
    auto b = a->sigmoid();
    b->backward();
    double s = 1.0 / (1.0 + std::exp(-1.0));
    ASSERT_NEAR(a->grad, s * (1.0 - s), 1e-6);
}

// ===== T14: 鏈式法則 =====

TEST(chain_rule) {
    auto a = Value::create(2.0);
    auto b = Value::create(3.0);
    auto c = Value::create(4.0);
    auto d = a * b;     // d = 6
    auto e = d + c;     // e = 10
    e->backward();
    assert(a->grad == 3.0);     // ∂e/∂a = b = 3
    assert(b->grad == 2.0);     // ∂e/∂b = a = 2
    assert(c->grad == 1.0);     // ∂e/∂c = 1
}

// ===== T15: 共用節點梯度累加 =====

TEST(shared_node_grad_accumulation) {
    auto a = Value::create(3.0);
    auto b = a + a;     // b = 2a = 6
    b->backward();
    assert(b->data == 6.0);
    assert(a->grad == 2.0);     // ∂(2a)/∂a = 2
}

// ===== T16: micrograd 官方範例 =====

TEST(micrograd_example) {
    auto a = Value::create(-4.0);
    auto b = Value::create(2.0);

    auto c = a + b;                                           // c = -2
    auto d = (a * b) + b->pow_val(3.0);                      // d = -8 + 8 = 0
    c = c + (c + Value::create(1.0));                         // c = -2 + (-2+1) = -3
    c = c + (Value::create(1.0) + c + a->neg());              // c = -3 + (1+(-3)) + 4 = -1

    auto d2 = d * Value::create(2.0);
    auto ba_relu = (b + a)->relu();
    d = d + (d2 + ba_relu);                                   // d = 0 + (0 + relu(-2)) = 0

    auto d3 = Value::create(3.0) * d;
    auto bna_relu = (b + a->neg())->relu();
    d = d + (d3 + bna_relu);                                  // d = 0 + (0 + relu(6)) = 6

    auto e = c - d;                                           // e = -1 - 6 = -7
    auto f = e->pow_val(2.0);                                 // f = 49
    auto g = f / Value::create(2.0);                          // g = 24.5
    g = g + (Value::create(10.0) / f);                        // g ≈ 24.7041

    ASSERT_NEAR(g->data, 24.7041, 1e-3);
    g->backward();
    ASSERT_NEAR(a->grad, 138.8338, 1e-3);
    ASSERT_NEAR(b->grad, 645.5773, 1e-3);
}

// ===== Main =====

int main() {
    std::cout << "\n🧪 cpp-grad tests\n" << std::endl;
    // 測試已經在全域建構時跑完了
    std::cout << "\n✅ " << tests_passed << "/" << tests_total << " passed\n" << std::endl;
    return (tests_passed == tests_total) ? 0 : 1;
}
