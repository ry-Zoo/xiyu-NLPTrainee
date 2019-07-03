def uniqueFirstPos(raw_str):
	dic={}
	if len(raw_str)==1:
		print("0")
	else:
		for index,element in enumerate(raw_str):
			if element in dic:
				dic[element].append(index)
				
			else:
				dic[element]=[index]
	min_index=len(raw_str)
	unique_char=""
	for key,value in dic.items():
		if len(value)>1:
			continue
		else:
			if value[0]<min_index:
				min_index=value[0]
				unique_char=str(key)
			else:
				continue
	print (str(unique_char)+":"+str(min_index))
	


raw_str="JINNAEJIB"
uniqueFirstPos(raw_str)