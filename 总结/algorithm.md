### 1. 0-1背包与完全背包

背包问题是一种组合优化的 NP 完全问题。

#### 0-1 背包问题

给定`n`个重量为$w_1, w_2, w_3,..., w_n$，价值为$v_1, v_2, v_3,...,v_n$的物品和容量为$C$的背包，求这些物品中最有价值的一个子集，使得在满足背包容量的前提下，包内的总价值最大。

如果限定每个物品只能选择0个或1个，则问题称为**0-1背包问题**。

利用动态规划求解0-1背包问题：

**状态定义**：定义`dp[i][j]`为背包容量大小为`j`时，由前`i`个物品能构成的最大价值。

**状态转换**：
```cpp
dp[i][j] = dp[i - 1][j] if j < w[i]
dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]) if j >= w[i]
```

**完整代码**：
```cpp
int knapsack(vector<int>& w, vector<int>& v, int C){
    int N = w.size();
    vector<vector<int>> dp(N + 1, vector<int>(C + 1, 0));
    for(int i = 1; i <= N; ++i){
        for(int j = 1; j <= C; ++j){
            if(j < w[i - 1]) dp[i][j] = dp[i - 1][j];
            else dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i - 1]] + v[i - 1]);
        }
    }
    return dp[N][C];
}
```

#### Leetcode实例：分割等和子集
给定一个**只包含正整数的非空数组**。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**思路**：由于数组中的每个元素只能选择0次或者1次，因此可以看做01背包问题。但这里只需要判断能否子集之和是否为总和的一半，因此状态转换方程会有所不同。

**完整代码**：
```cpp
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if(sum & 1) return false;
        int target = sum / 2;
        vector<vector<bool>> dp(nums.size() + 1, vector<bool>(target + 1, false));
        for(int i = 1; i <= nums.size(); ++i){
            for(int j = 1; j <= target; ++j){
                if(j == nums[i - 1]) dp[i][j] = true;
                else if(j < nums[i - 1]) dp[i][j] = dp[i - 1][j];
                else dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
            }
            if(dp[i][target]) return true;
        }
        return dp[nums.size()][target];
    }
};
```

#### 完全背包问题

完全背包问题与01背包的不同之处在于不限制每种物品的个数。

完全背包问题也可以用动态规划求解。

**状态定义**：定义`dp[i][j]`为背包容量大小为`j`时，由前`i`个物品能构成的最大价值。

**状态转换**：
```cpp
dp[i][j] = max(dp[i - 1][j - k * w[i]] + k * v[i]) for k * w[i] <= j
```
**完整代码**：
```cpp
int knapsack(vector<int>& w, vector<int>& v, int C){
    int N = w.size();
    vector<vector<int>> dp(N + 1, vector<int>(C + 1, 0));
    for(int i = 1; i <= N; ++i){
        for(int j = 1; j <= C; ++j){
            for(int k = 0; k * w[i - 1] <= j; ++k){
                dp[i][j] = max(dp[i][j], dp[i][j - k * w[i - 1]] + v[i - 1]);
            }
        }
    }
    return dp[N][c];
}
```

#### Leetcode实例：零钱兑换II

给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 

**完整代码**：
```cpp
class Solution{
public:
    int change(int amount, vector<int>& coins){
        int N = coins.size(), C = amount;
        vector<vector<int>> dp(N + 1, vector<int>(C + 1, 0));
        for(int i = 0; i <= N; ++i) dp[i][0] = 1;
        for(int i = 1; i <= N; ++i){
            for(int j = 1; j <= C; ++j){
                for(int k = 0; k * coins[i - 1] <= j; ++k){
                    dp[i][j] += dp[i - 1][j - k * coins[i - 1]];
                }
            }
        }
        return dp[N][C];
    }
};
```
