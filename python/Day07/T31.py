def distinct_substr(raw_str):
	res_ls=[]
	cur_ls=[]
	if len(raw_str)==0:
		return False	
	else:	
		for i in range(0,len(raw_str)):
			if raw_str[i] in cur_ls:
				if len(res_ls)<len(cur_ls):
					res_ls=cur_ls
					cur_ls=[]
				else:
					cur_ls=[]
				cur_ls.append(raw_str[i])
			else:
				cur_ls.append(raw_str[i])

			
		return res_ls

res_str=distinct_substr("abcabcbb")
print(res_str)
print(len(res_str))