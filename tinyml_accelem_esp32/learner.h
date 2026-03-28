/*
 * learner.h — On-Device SGD (обучение на устройстве)
 *
 * Ответственность:
 *   - Forward Pass (предсказание через head_W, head_B)
 *   - Backward Pass (SGD с градиентным клиппингом)
 *   - Кольцевой буфер задержки для Self-Supervised Learning
 */
#ifndef LEARNER_H
#define LEARNER_H

#include "config.h"

// --- Прототипы весов (определены в head_weights.h) ---
extern float head_W[EMBEDDING_SIZE][NUM_PREDICTIONS];
extern float head_B[NUM_PREDICTIONS];

// --- Кольцевой буфер задержки ---
static float  delay_emb[DELAY_TICKS][EMBEDDING_SIZE];
static float  delay_pred[DELAY_TICKS][NUM_PREDICTIONS];
static bool   delay_ok[DELAY_TICKS] = {false};
static int    delay_idx = 0;

// ─────────────── Forward Pass ───────────────
// pred = head_W^T · embedding + head_B
inline void head_forward(const float* embedding, float out[NUM_PREDICTIONS]) {
    for (int j = 0; j < NUM_PREDICTIONS; j++) {
        float sum = head_B[j];
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            sum += embedding[i] * head_W[i][j];
        }
        out[j] = sum;
    }
}

// ─────────────── Backward Pass (SGD) ───────────────
// Обновляет head_W и head_B на основе MSE-ошибки между старым
// предсказанием и реальным значением (ground truth).
// Возвращает суммарный MSE loss (для телеметрии/логов).
inline float head_backward(const float* old_emb,
                            const float* old_pred,
                            const float  gt[NUM_PREDICTIONS]) {
    float total_loss = 0.0f;

    for (int j = 0; j < NUM_PREDICTIONS; j++) {
        float err = old_pred[j] - gt[j];

        // Клиппинг градиента
        if (err >  GRADIENT_CLIP) err =  GRADIENT_CLIP;
        if (err < -GRADIENT_CLIP) err = -GRADIENT_CLIP;

        total_loss += err * err;  // MSE

        // dL/dW_ij = err * emb_i
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            head_W[i][j] -= LEARNING_RATE * err * old_emb[i];
        }
        head_B[j] -= LEARNING_RATE * err;
    }

    return total_loss / NUM_PREDICTIONS;  // Средний MSE по осям
}

// ─────────────── Цикл обучения с буфером задержки ───────────────
// Выполняет backward (если буфер заполнен) + forward + сохранение.
// Возвращает текущий loss (или -1, если backward не вызывался).
inline float learner_step(const float* embedding,
                           const float  gt[NUM_PREDICTIONS],
                           float        pred_out[NUM_PREDICTIONS]) {
    float loss = -1.0f;

    // 1. Backward: учимся на старых данных
    if (delay_ok[delay_idx]) {
        loss = head_backward(delay_emb[delay_idx],
                             delay_pred[delay_idx], gt);
    }

    // 2. Forward: предсказываем
    head_forward(embedding, pred_out);

    // 3. Сохраняем в буфер для будущего обучения
    memcpy(delay_emb[delay_idx],  embedding, sizeof(float) * EMBEDDING_SIZE);
    memcpy(delay_pred[delay_idx], pred_out,  sizeof(float) * NUM_PREDICTIONS);
    delay_ok[delay_idx] = true;

    delay_idx = (delay_idx + 1) % DELAY_TICKS;
    return loss;
}

#endif // LEARNER_H
