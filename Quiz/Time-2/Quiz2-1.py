import re
#test
def Caculate(raw_str):
	raw_str=raw_str.strip()
	if len(raw_str)==0:
		return False
	first_str=re.split(r"[+-]",raw_str)
	first_sim=[s for s in raw_str if s =="+" or s=="-"]
	first_enum=[]
	#print(str(first_sim))
	for i in range(0,len(first_str)):
		if "*" in first_str[i] or "/" in first_str[i]:
			second_str=re.split(r"[*/]",first_str[i])
			second_sim=[s for s in first_str[i] if s =="*" or s=="/"]
			second_ls=[int(x) for x in second_str]
			sec_res=second_ls[0]
			#print(second_sim)
			for j in range(0,len(second_sim)):
				if second_sim[j]=="*":
					sec_res=sec_res*second_ls[j+1]
				elif second_sim[j]=="/":
					sec_res=sec_res/second_ls[j+1]
				else:
					return 0
			#print(sec_res)
			first_enum.append(sec_res)
			#print(first_enum)
		else:
			first_enum.append(int(first_str[i]))
	
	res=0

	for j in range(0,len(first_sim)):
		if first_sim[j]=="+":
			res=first_enum[j]+first_enum[j+1]
		elif first_sim[j]=="-":
			res=first_enum[j]-first_enum[j+1]
		else:
			return 0
	#print(res)
	return res


Caculate("3+2*2*2")