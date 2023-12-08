import numpy as np

epsilon = 1e-3


def compute_transition_matrix(model):
    M, N = model.M, model.N
    P = np.zeros((M, N, 4, M, N))
    for r in range(M) :
        for c in range(N) :
            for a in range(4) :
                for r_prime in range(M):
                    for c_prime in range(N) :
                        if model.T[r, c] :
                            P[r, c, a, r_prime, c_prime] = 0
                        else :
                            # (r_want, c_want) : 원하는 방향으로 이동했을 때의 좌표
                            # (r_cntclk, c_cntclk) : 원하는 방향의 반시계방향으로 이동했을 때의 좌표
                            # (r_clk, c_clk) : 원하는 방향의 시계방향으로 이동했을 때의 좌표
                            if a == 0 : # left(0)
                                r_want, c_want = r, max(c-1, 0) 
                                r_cntclk, c_cntclk = min(r+1, M-1), c
                                r_clk, c_clk = max(r-1, 0), c
                            elif a == 1 : # up(1)
                                r_want, c_want = max(r-1, 0), c
                                r_cntclk, c_cntclk = r, max(c-1, 0)
                                r_clk, c_clk = r, min(c+1, N-1)
                            elif a == 2 : # right(2)
                                r_want, c_want = r, min(c+1, N-1)
                                r_cntclk, c_cntclk = max(r-1, 0), c
                                r_clk, c_clk = min(r+1, M-1), c
                            elif a == 3 : # down(3)
                                r_want, c_want = min(r+1, M-1), c
                                r_cntclk, c_cntclk = r, min(c+1, N-1)
                                r_clk, c_clk = r, max(c-1, 0)
                            
                            # want, cntclk, clk 경우마다 이동할 좌표가 벽인지 확인. 벽이면 현재 위치로 바꿈.
                            if model.W[r_want, c_want] :
                                r_want, c_want = r, c
                            if model.W[r_cntclk, c_cntclk] :
                                r_cntclk, c_cntclk = r, c
                            if model.W[r_clk, c_clk] :
                                r_clk, c_clk = r, c
                            
                            # r_prime, c_prime이 want, cntclk, clk인 경우 각각 D[r, c, 0] D[r, c, 1] D[r, c, 2]의 확률을 가진다.
                            if r_want == r_prime and c_want == c_prime : 
                                P[r, c, a, r_prime, c_prime] += model.D[r, c, 0]
                            if r_cntclk == r_prime and c_cntclk == c_prime :
                                P[r, c, a, r_prime, c_prime] += model.D[r, c, 1]
                            if r_clk == r_prime and c_clk == c_prime :
                                P[r, c, a, r_prime, c_prime] += model.D[r, c, 2]
                            
    return P


def update_utility(model, P, U_current):
    M, N = model.M, model.N
    U_next = np.zeros((M, N))
    for r in range(M) :
        for c in range(N) :
            # U_(i+1)(s) = R(s) + ~~~
            U_next[r, c] += model.R[r, c]
            max_sumU = -np.inf
            for a in range(4) :
                sumU = 0
                for r_prime in range(M) :
                    for c_prime in range(N) :
                        # sum of (P[s'|s, a] * U_i(s')) for all s'.
                        sumU += P[r, c, a, r_prime, c_prime] * U_current[r_prime, c_prime]
                # 모든 action에 대해 가장 큰 sumU를 구한다.
                max_sumU = max(max_sumU, sumU)
            # U_(i+1)(s) = R(s) + gamma * max_sumU
            U_next[r, c] += model.gamma * max_sumU
    
    return U_next


def value_iteration(model):
    P = compute_transition_matrix(model)
    M, N = model.M, model.N
    U_current = np.zeros((M, N))
    U_next = np.zeros((M, N))

    # 최대 100번 iterate
    for i in range(100) :
        # update utility
        U_next = update_utility(model, P, U_current)
        # 모든 state에 대해 U_next와 U_current의 차이가 epsilon보다 작은 경우 result = True
        result = calculate_delta(model, U_current, U_next)
        if result : break
        # next를 current로 넘기고 다시 반복
        U_current = U_next.copy()
    return U_next

def calculate_delta(model, U_current, U_next) :
    M, N = model.M, model.N
    for r in range(M) :
        for c in range(N) :
            # epsilon보다 커지는 state 있으면 False return.
            if abs(U_current[r, c] - U_next[r, c]) > epsilon :
                return False
    return True

    