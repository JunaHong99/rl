# SAC 알파 붕괴(α-collapse) 근본 원인 분석 리포트

**대상 코드**: `sac_trainer.py`, `gnn_policy.py` (Phase 3.3-SAC)
**증상**: 학습 중 엔트로피 온도 $\alpha$가 초기값 $e^{-1}\approx0.37$에서 $4\times10^{-33}$까지 단조 붕괴 → 탐색 소실 → frozen policy
**결론**: **`action_scale`(≈$10^{-3}$)이 적용된 action을 그대로 Q-critic에 입력하고 있어 $\partial Q/\partial a \approx 0$이 되는 것이 근본 원인.**

---

## 1. 요약 (TL;DR)

| 항목 | 판정 |
|---|---|
| `log_alpha` 파라미터화 (`exp` 사용) | ✅ 정상 |
| Alpha loss 부호 / `detach()` | ✅ 정상 |
| `log_prob`의 action-dim 합산 (`sum(dim=-1)`) | ✅ 정상 |
| tanh Jacobian 보정 | ✅ 정상 |
| **Q-critic의 action 입력 스케일** | ❌ **근본 원인** |
| `target_entropy = -6.0` | ⚠️ 과도하게 공격적 |
| `ReplayBuffer(num_envs=None)` | ❌ 별도 크래시 버그 |

알파는 **버그로 터진 것이 아니라, 피드백 컨트롤러로서 정직하게 동작한 결과**다. 정책 엔트로피가 단 한 번도 타겟 아래로 내려가지 못했기 때문에 알파가 영원히 감소했을 뿐이다. 진짜 문제는 **엔트로피를 줄일 유일한 힘인 $Q$ 그래디언트가 죽어 있다는 것**이다.

---

## 2. 알파 업데이트의 정상 동작 이해

자동 엔트로피 조절의 목적함수는

$$J(\alpha) = \mathbb{E}_{a_t \sim \pi_t}\Big[-\alpha\big(\log \pi_t(a_t \mid s_t) + \bar{\mathcal{H}}\big)\Big]$$

이고, `log_alpha`에 대한 기울기는

$$\nabla_{\log\alpha} J = -\alpha\big(\log\pi(a|s) + \bar{\mathcal{H}}\big)$$

즉 알파는 다음 조건에서만 **반등**한다.

$$\log\pi(a|s) > -\bar{\mathcal{H}} = 6 \qquad \Longleftrightarrow \qquad \mathcal{H}(\pi) < \bar{\mathcal{H}} = -6$$

현재 코드에서는 이 조건이 **한 번도 충족되지 않는다.** 따라서 그래디언트 부호가 항상 음수로 고정된다.

### 2.1 붕괴 속도의 정량적 검증

Adam은 그래디언트 크기를 정규화하므로, 부호만 일정하면 스텝 크기는 거의 항상 학습률과 같다.

$$\Delta \log\alpha \;\approx\; -\eta_\alpha \times \texttt{updates\_per\_step} \;=\; -3\times10^{-4} \times 4 \;=\; -1.2\times10^{-3} \ \ \text{per env step}$$

관측된 최종값 $\alpha = 4\times10^{-33}$ 은 $\log\alpha \approx -75$ 에 해당하므로, 초기값 $\log\alpha = -1$ 에서 필요한 스텝 수는

$$\frac{-1 - (-75)}{1.2\times10^{-3}} \;\approx\; 6.2\times10^{4}\ \text{env steps}$$

> **검증 포인트**: 학습 로그의 $\log\alpha$ 곡선이 **거의 완벽한 직선**이었다면 이 진단이 확정된다. (지수적 감쇠나 진동이 아니라 로그 공간의 선형 하강)

---

## 3. 근본 원인: Q가 action을 물리적으로 볼 수 없다

### 3.1 문제 코드

`gnn_policy.py` — `GNNActor`:

```python
action_scale = torch.tensor([0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005])
...
action = self.action_scale * action_norm     # 크기 ~1e-3
return action, log_prob, None                # ← 이 스케일된 값이 버퍼/Q로 흘러감
```

`gnn_policy.py` — `GNNQCritic.forward`:

```python
h = torch.cat([rod_raw, batch.u, rod_mp, u, action], dim=-1)
#              O(1)     O(1)     O(1)    O(1)  ~1e-3
#              (32)     (4)      (512)   (512)  (6)   → head_in ≈ 1066
```

### 3.2 두 겹의 그래디언트 소실

**(a) 입력 단계 — action이 다른 특징에 묻힌다.**

Q head 첫 레이어에서 action 항의 기여는 $w \cdot 10^{-3}$ 수준이다. 나머지 1060개 입력이 $O(1)$ 크기이므로, action은 **1066차원 중 6차원, 그마저도 1000배 작은 크기**로 들어간다. 학습 초기 Q는 사실상

$$Q(s,a) \;\approx\; Q(s)$$

가 된다. 더 나쁜 것은, 해당 가중치의 그래디언트 역시 입력 크기에 비례하므로 **action-dependence 학습 자체가 $10^3$배 느리다.**

**(b) 역전파 단계 — chain rule에서 한 번 더 죽는다.**

$$\frac{\partial Q}{\partial u} \;=\; \underbrace{\frac{\partial Q}{\partial a}}_{\approx\,0} \cdot \underbrace{\texttt{action\_scale}}_{10^{-3}} \cdot \big(1 - \tanh^2 u\big)$$

즉 actor loss

$$\mathcal{L}_\pi = \mathbb{E}\big[\alpha \log\pi(a|s) - \textstyle\min_i Q_i(s, a)\big]$$

에서 **$Q$ 항의 그래디언트가 엔트로피 항 대비 약 $10^{-3}$배로 억제된다.**

### 3.3 자기강화 붕괴 루프

```
  ①  ∂Q/∂a ≈ 0
        ↓
  ②  log_std를 줄일 유일한 압력(=Q 항)이 소실
        ↓
  ③  남은 것은 -α·log π 항뿐 → 엔트로피를 오히려 "최대화"
      → log_std가 LOG_STD_MAX(=2) 방향으로 밀림
        ↓
  ④  log π가 계속 음수 → ∇_logα = -(log π + 6) 부호 고정(음)
      → α가 로그 공간에서 직선 하강
        ↓
  ⑤  α ≈ 0 → 엔트로피 항마저 소멸
      → 정책을 움직이는 힘이 아무것도 없음
        ↓
      ★ FROZEN POLICY (관측된 증상)
```

②와 ④가 서로를 강화하는 구조라 **한 번 진입하면 자력 복구가 불가능**하다.

---

## 4. 수정안

### 4.1 핵심 수정 — 학습 파이프라인을 정규화 action으로 통일

**원칙: `action_scale`은 환경에 나갈 때만 곱한다.** 버퍼, Q-critic, actor loss는 전부 $a_{\text{norm}} = \tanh(u) \in (-1,1)^6$ 로 통일한다.

`gnn_policy.py` — `GNNActor`:

```python
def get_action_and_log_prob(self, batch, deterministic: bool = False):
    """정규화된 action(-1,1)과 log-prob 반환. 스케일링은 to_env_action에서."""
    mean_raw, log_std = self.forward(batch)

    if deterministic:
        action_norm = torch.tanh(mean_raw)
        return action_norm, None, None          # ← 정규화된 값

    std = torch.exp(log_std)
    dist = Normal(mean_raw, std)
    u = dist.rsample()                          # reparameterized
    action_norm = torch.tanh(u)
    log_prob = dist.log_prob(u) - torch.log(1.0 - action_norm.pow(2) + self.EPS)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return action_norm, log_prob, None          # ← Q/buffer가 소비하는 값

def to_env_action(self, action_norm):
    """환경에 나갈 때만 물리 스케일 적용."""
    return self.action_scale * action_norm
```

롤아웃 루프:

```python
a_norm, logp, _ = agent.actor.get_action_and_log_prob(state)
env_action = agent.actor.to_env_action(a_norm)          # 환경엔 스케일된 값
next_state, r, done, _ = env.step(env_action)
buffer.add_batch(state, a_norm, r, next_state, done)    # ★ 버퍼엔 정규화된 값
```

warmup 구간의 random action도 동일하게 $\mathcal{U}(-1,1)^6$ 에서 뽑아 정규화 좌표로 저장해야 한다.

> `sac_trainer.py`는 **수정 불필요**. 버퍼에 저장되는 좌표계만 바뀌면 $Q(s, a_{\text{norm}})$ 가 되고 actor loss의 그래디언트 경로도 자동으로 살아난다.

### 4.2 `target_entropy` 완화

tanh-squashed 정책은 $(-1,1)^6$ 에 갇혀 있으므로 엔트로피 상한이 유한하다.

$$\mathcal{H}_{\max} = 6\log 2 \approx 4.16 \ \text{nats}$$

$\bar{\mathcal{H}} = -6$ 은 각 차원 평균 $-1$ nat, 즉 **거의 결정론적 정책**을 요구한다. 탐색이 목적이라면:

$$\bar{\mathcal{H}} = -0.5 \cdot \dim(\mathcal{A}) = -3$$

```python
target_entropy: float = -3.0   # -6.0 → -3.0 (0.5 × action_dim)
```

### 4.3 부수 버그 — `ReplayBuffer(num_envs=None)`

`sac_trainer.py`:

```python
self.buffer = ReplayBuffer(
    capacity=cfg.buffer_size,
    num_envs=None,        # ← 주석은 "set when first added"지만 설정 코드가 없음
    action_dim=6,
    device=device,
)
```

`add_batch`는 `B = self.num_envs` 를 그대로 사용하므로 `x.view(None, N, -1)` 에서 크래시한다. 외부에서 `trainer.buffer.num_envs = N` 을 대입하고 있지 않다면 반드시 수정할 것. 아울러 `action_dim=6` 하드코딩은 `cfg.target_entropy` 와 따로 놀 여지가 있으므로 `agent.actor.action_dim` 에서 받아오는 것이 안전하다.

### 4.4 `alpha_min` floor — 유지 권장

```python
alpha_min: float = 0.01
```

Adam 특성상 $\log\alpha$ 가 $-75$ 까지 내려가면 그래디언트 부호가 뒤집혀도 복구에 **25만 스텝**이 걸린다. floor는 이 비가역성을 막는 안전장치로 유효하다. 다만 이는 **증상 억제일 뿐이며**, 4.1 수정 후에는 floor에 닿지 않는 것이 정상이다.

### 4.5 리워드 스케일 점검

알파의 균형점은 $Q$ 스케일에 **상대적**이다. 로그의 `q1_mean` 이 수백 단위라면 $\alpha \cdot \log\pi$ 항은 여전히 무시된다. 리워드 정규화 또는 스케일 축소를 검토할 것.

---

## 5. 검증 절차

### 5.1 즉시 확인 — $\partial Q/\partial a$ 프로브

`SACTrainer.update()` 안에 삽입:

```python
a_probe = a_new.detach().requires_grad_(True)
q1p, q2p = self.agent.q(s, a_probe)
g = torch.autograd.grad(torch.min(q1p, q2p).sum(), a_probe)[0]
dq_da = g.norm(dim=-1).mean().item()
```

| 값 | 해석 |
|---|---|
| $\lesssim 10^{-4}$ | **Q가 action을 무시 중 — 근본 원인 확정** |
| $O(1)$ | Q 그래디언트 정상, 다른 원인 탐색 필요 |

수정 전에는 $10^{-4}$ 이하, 수정 후 $O(1)$ 로 올라오면 성공이다.

### 5.2 지속 모니터링 — 로깅 추가

```python
return {
    ...,
    "entropy":       -log_pi_new_flat.mean().item(),      # ★ target_entropy와 같은 그래프에
    "target_entropy": cfg.target_entropy,
    "log_std_mean":   self.agent.actor.log_std.mean().item(),
    "dQ_da":          dq_da,
    "alpha":          self.alpha.item(),
}
```

### 5.3 판독 기준

| 관측 패턴 | 진단 |
|---|---|
| 엔트로피가 계속 타겟 **위**에 있고 `log_std`가 상한 근처에 고정, `dQ_da` ≈ 0 | **본 리포트의 근본 원인 확정** |
| 엔트로피가 타겟 **아래**로 갔는데 알파가 안 오름 | 트레이너 버그 (현 코드에는 해당 없음) |
| `dQ_da`는 $O(1)$인데 알파가 여전히 하강 | `target_entropy` 과도 (§4.2) 또는 리워드 스케일 (§4.5) |
| 엔트로피가 타겟 근처에서 **수렴/진동** | ✅ 컨트롤러 정상 작동 |

**성공 기준**: 알파가 단조 하강이 아니라 **하강 후 반등하여 특정 값 근처에서 진동**하고, 엔트로피가 $\bar{\mathcal{H}}$ 근처로 수렴할 것.

---

## 6. 수정 우선순위

| 순위 | 항목 | 위치 | 영향 |
|---|---|---|---|
| **P0** | 버퍼/Q에 정규화 action 사용 | `gnn_policy.py` + 롤아웃 루프 | 근본 원인 |
| **P0** | `ReplayBuffer(num_envs=None)` | `sac_trainer.py` | 크래시 |
| **P1** | `target_entropy: -6.0 → -3.0` | `SACConfig` | 탐색량 |
| **P1** | `dQ_da` / `entropy` 로깅 | `SACTrainer.update` | 검증 |
| **P2** | 리워드 스케일 정규화 | 환경 | 알파 균형점 |
| **P2** | `action_dim` 하드코딩 제거 | `SACTrainer.__init__` | 유지보수 |

---

## 부록 A. 무죄 판정 항목 (재확인 불필요)

다음은 흔한 SAC 버그이나, 현 코드에서는 **정상 구현**되어 있음을 확인했다.

**A.1 `log_alpha` 파라미터화** — 알파를 직접 최적화하면 음수로 발산하다 clamp에 박혀 그래디언트가 죽는다. 현 코드는 $\alpha = \exp(\log\alpha)$ 로 올바르게 처리.

**A.2 Alpha loss 부호와 detach**
```python
alpha_loss = -(self.log_alpha * (log_pi_new_flat.detach() + cfg.target_entropy)).mean()
```
부호, `detach()` 위치 모두 정확. `log_pi`를 detach하지 않으면 알파 그래디언트가 정책으로 새어들어가 서로 밀어내는데, 그런 문제 없음.

**A.3 tanh Jacobian 보정 + 차원 합산**
```python
log_prob = dist.log_prob(u)
log_prob = log_prob - torch.log(1.0 - action_norm.pow(2) + self.EPS)
log_prob = log_prob.sum(dim=-1, keepdim=True)
```
정확히 다음을 구현하고 있다.

$$\log\pi(a|s) = \sum_{i=1}^{6}\Big[\log\mathcal{N}(u_i;\mu_i,\sigma_i) - \log\big(1-\tanh^2(u_i)+\epsilon\big)\Big]$$

`.mean(-1)` 오용이나 보정항 누락 없음. (이 둘은 알파 붕괴의 가장 흔한 원인이지만 여기선 해당 없음)

**A.4 `log_std` clamp** — `LOG_STD_MIN=-5, LOG_STD_MAX=2`로 클램프되어 있어 tanh 포화 구간 발산 없음.

**A.5 Target network 및 그래디언트 격리** — `soft_update_target`의 Polyak 갱신 정확. `actor_loss.backward()`가 Q 파라미터에 그래디언트를 누적시키지만, 다음 iteration의 `q_opt.zero_grad()`가 `q_loss.backward()` 직전에 호출되므로 오염 없음.
