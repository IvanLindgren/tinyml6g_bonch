/*
 * learner.h — On-Device Adam Learning (обучение на устройстве)
 *
 * Ответственность:
 *   - Forward Pass (предсказание через head_W, head_B)
 *   - Backward Pass (Adam optimizer)
 *   - Кольцевой буфер задержки для Self-Supervised Learning
 */
#ifndef LEARNER_H
#define LEARNER_H

#include "config.h"
#include <math.h>
#include <Arduino.h>

// --- Прототипы весов (определены в head_weights.h) ---
extern float head_W[EMBEDDING_SIZE][NUM_PREDICTIONS];
extern float head_B[NUM_PREDICTIONS];

// --- Кольцевой буфер задержки ---
static float  delay_emb[DELAY_TICKS][EMBEDDING_SIZE];
static float  delay_pred[DELAY_TICKS][NUM_PREDICTIONS];
static bool   delay_ok[DELAY_TICKS] = {false};
static int    delay_idx = 0;

// --- Состояние оптимизатора Adam ---
static float m_W[EMBEDDING_SIZE][NUM_PREDICTIONS] = {0};
static float v_W[EMBEDDING_SIZE][NUM_PREDICTIONS] = {0};
static float m_B[NUM_PREDICTIONS] = {0};
static float v_B[NUM_PREDICTIONS] = {0};

static float beta1 = 0.9f;
static float beta2 = 0.999f;
static float epsilon = 1e-8f;
static float beta1Decayed = 1.0f;
static float beta2Decayed = 1.0f;

static float current_lr = LEARNING_RATE;

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

// ─────────────── Backward Pass (Adam) ───────────────
// Обновляет head_W и head_B на основе MSE-ошибки с Adam.
inline float head_backward(const float* old_emb,
                            const float* old_pred,
                            const float  gt[NUM_PREDICTIONS]) {
    float total_loss = 0.0f;

    beta1Decayed *= beta1;
    beta2Decayed *= beta2;

    for (int j = 0; j < NUM_PREDICTIONS; j++) {
        float err = old_pred[j] - gt[j];

        // Клиппинг градиента
        if (err >  GRADIENT_CLIP) err =  GRADIENT_CLIP;
        if (err < -GRADIENT_CLIP) err = -GRADIENT_CLIP;

        total_loss += err * err;  // MSE

        // dL/dW_ij = err * emb_i
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            float g = err * old_emb[i];
            m_W[i][j] = beta1 * m_W[i][j] + (1.0f - beta1) * g;
            v_W[i][j] = beta2 * v_W[i][j] + (1.0f - beta2) * g * g;
            
            float mhat = m_W[i][j] / (1.0f - beta1Decayed);
            float vhat = v_W[i][j] / (1.0f - beta2Decayed);
            
            head_W[i][j] -= current_lr * mhat / (sqrtf(vhat) + epsilon);
        }
        
        // dL/dB_j = err
        float g_b = err;
        m_B[j] = beta1 * m_B[j] + (1.0f - beta1) * g_b;
        v_B[j] = beta2 * v_B[j] + (1.0f - beta2) * g_b * g_b;
        
        float mhat_b = m_B[j] / (1.0f - beta1Decayed);
        float vhat_b = v_B[j] / (1.0f - beta2Decayed);
        
        head_B[j] -= current_lr * mhat_b / (sqrtf(vhat_b) + epsilon);
    }

    return total_loss / NUM_PREDICTIONS;
}

// ─────────────── Цикл обучения с буфером задержки ───────────────
// Выполняет backward (если буфер заполнен) + forward + сохранение.
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
    for (int i=0; i<EMBEDDING_SIZE; i++) delay_emb[delay_idx][i] = embedding[i];
    for (int j=0; j<NUM_PREDICTIONS; j++) delay_pred[delay_idx][j] = pred_out[j];
    
    delay_ok[delay_idx] = true;
    delay_idx = (delay_idx + 1) % DELAY_TICKS;
    
    // 4. Learning Rate Decay (каждые 6000 шагов = ~1 минута)
    static int steps = 0;
    if (++steps >= 6000) {
        steps = 0;
        current_lr *= 0.99f; // Умножаем на 0.99
        if (current_lr < 0.0001f) current_lr = 0.0001f; // Ограничитель снизу
    }
    
    return loss;
}

#endif // LEARNER_H
