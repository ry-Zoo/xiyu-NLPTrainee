

def LongestArea(raw_str):
	dec_str="B"+raw_str
	n=len(dec_str)
	if n<2:
		return  0
	ans = 0

	dp=[0]*n

	for i in range(1,n):
		if dec_str[i]==")":
			if dec_str[i-1-dp[i-1]]=="(":
				dp[i]=dp[i-1]+2
			dp[i]+=dp[i-dp[i]]
		ans= max(ans,dp[i])
	print (str(ans))

LongestArea(")()())")