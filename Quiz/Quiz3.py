class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        if len(s)==0 or len(t)==0:
            return False
        else:
            dp=[[0 for i in range(0,len(t)+1)] for j in range(0,len(s)+1)]
            for i in range(0,len(s)+1):
                dp[i][0]=1
            for j in range(1,len(t)+1):
                for i in range(1,len(s)+1):
                    if s[i-1]==t[j-1]:
                        dp[i][j]=dp[i-1][j-1]+dp[i-1][j]
                    else:
                        dp[i][j]=dp[i-1][j]
            return dp[len(s)][len(t)]

                        