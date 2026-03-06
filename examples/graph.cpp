// cpp-grad 計算圖視覺化範例
// 產生 graphviz DOT 檔，像 micrograd 的 trace_graph
#include "cpp_grad.hpp"

int main() {
    // 簡單的神經元：y = relu(w1*x1 + w2*x2 + b)
    auto x1 = Value::create(2.0)->label("x1");
    auto x2 = Value::create(0.0)->label("x2");
    auto w1 = Value::create(-3.0)->label("w1");
    auto w2 = Value::create(1.0)->label("w2");
    auto b  = Value::create(6.88)->label("b");

    auto x1w1 = (x1 * w1)->label("x1*w1");
    auto x2w2 = (x2 * w2)->label("x2*w2");
    auto sum  = (x1w1 + x2w2)->label("sum");
    auto n    = (sum + b)->label("n");
    auto o    = n->tanh_val()->label("o");

    o->backward();

    // 印出計算圖
    std::cout << *o << std::endl;

    // 產生 graphviz 圖
    o->draw_dot("neuron");

    return 0;
}
