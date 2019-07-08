def FindCYear(raw_str,Num_dict):
	result=""
	for i in range(0,len(raw_str)):
		if raw_str[i]=='年':
			year_str=raw_str[i-4:i]
			year=CYear2Num(year_str,Num_dict)
			if year!="":
				result=raw_str[:i-3]+year+raw_str[i:]
		else:
			continue
	print(result)
def CYear2Num(year_str,Num_dict):
	year_num=[]
	res_str=""
	for y in year_str:
		if y in Num_dict.keys():
			year_num.append(Num_dict[y])
		else:
			break
	if len(year_num)>0:
		res_str=''.join(year_num)
		return res_str
	else:
		return ""


#raw_str="在一九四九年新中国成立"
raw_str="比一九九零年低百分之五点二"
Num_dict={
	"零":"0",
	"一":"1",
	"二":"2",
	"三":"3",
	"四":"4",
	"五":"5",
	"六":"6",
	"七":"7",
	"八":"8",
	"九":"9"
	}
FindCYear(raw_str,Num_dict)