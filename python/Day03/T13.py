def Is_cross_str(s1,s2,s3):
	length_1=len(s1)
	length_2=len(s2)
	length_3=len(s3)
	if length_1+length_2!=length_3:
		print('exception parameter')
	else:
		dp=[[False for i in range(0,len(s1))] for j in range(0,len(s2))]
		dp[0][0]=True
		
		for c in range(1,len(s1)+1):
			dp[c][0]=dp[c-1][0] and s1[c-1]==s3[c-1]
		for c in range(1,len(s2)+1):
			dp[0][c]=dp[0][c-1] and  s2[c-1]==s3[c-1]
		for i in range(1,len(s1)+1):
			for j in rangr(1,le(s2)+1):
				dp[i][j]=(dp[i][j-1] and s1[i-1]==s3[i+j-1]) or (dp[j-1][i] and  s2[j-1]==s3[i+j-1])
	print(str(dp[-1][-1]))
	

s1="acc"
s2="da"
s3="adacc"
Is_cross_str(s1,s2,s3)


	