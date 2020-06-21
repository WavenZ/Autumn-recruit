#include <bits/stdc++.h>
#include "test.h"
using namespace std;
int bb01(vector<int>& weight, vector<int>& val, int W){
    int N = weight.size();
    vector<vector<int>> dp(N + 1, vector<int>(W + 1));
    for(int i = 1; i <= N; ++i){
        for(int w = 1; w <= W; ++w){
            if(w < weight[i - 1]) dp[i][w] = dp[i - 1][w];
            else dp[i][w] = max(dp[i - 1][w],
                                dp[i - 1][w - weight[i - 1]] + val[i - 1]);
            cout << dp[i][w] << " ";
        }
        cout << endl;
    }

    return dp[N][W];
}

        vector<vector<char>> vec = {{}, {}, {'a', 'b', 'c'}, {'d', 'e', 'f'}, {'g', 'h', 'i'}, {'j', 'k', 'l'}, 
        {'m', 'n', 'o'}, {'p', 'q', 'r', 's'}, {'t', 'u', 'v'}, {'w', 'x', 'y', 'z'}};

int main(int argc, char* argv[]){
    vector<int> weight = {2, 3, 4, 5};
    vector<int> val = {3, 4, 5, 6};
    int W = 8;
    cout << bb01(weight, val, W) << endl;
    return 0;
}