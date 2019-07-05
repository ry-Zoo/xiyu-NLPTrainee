class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1)+len(s2)!=len(s3):
            return False
        else:
            dp=[[False for i in range(0,len(s1)+1)] for j in range(0,len(s2)+1)]
            dp[0][0]=True
            for c in range(1,len(s1)+1):
                dp[c][0]=(dp[c-1][0] and s1[c-1]==s3[c-1])
            for c in range(1,len(s2)+1):
                dp[0][c]=(dp[0][c-1] and s2[c-1]==s3[c-1])
            for i in range(1,len(s1)+1):
                for j in range(1,len(s2)+1):
                    dp[i][j] = (dp[i-1][j] and s1[i-1]==s3[i+j-1]) or (dp[i][j-1] and s2[j-1]==s3[i+j-1])
            return dp[-1][-1]