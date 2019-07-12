import re

def Caculate(raw_str):
	raw_str=raw_str.strip()
	if len(raw_str)==0:
		return False
	first_str=re.split(r"[+-]",raw_str)
	first_sim=[s for s in raw_str if s =="+" or s=="-"]
	first_enum=[]
	#print(str(first_sim))
	for i in range(0,len(first_str)):
		if "*" in first_str[i]:
			second_str=first_str[i].split("*")
			second_ls=[int(x) for x in second_str]
			first_enum.append(second_ls[0]*second_ls[1])
			#print(first_enum)
		elif "/" in first_str[i]:
			second_str=first_str[i].split("/")
			second_ls=[int(x) for x in second_str]
			first_enum.append(second_ls[0]/second_ls[1])
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

	return res


Caculate("3+2*2")