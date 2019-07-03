def TeleNum_Comb(Num_str):
	res_ls=[]
	num_char_map={
	"2":"abc",
	"3":"def",
	"4":"ghi",
	"5":"jkl",
	"6":"mno",
	"7":"pqrs",
	"8":"tuv",
	"9":"wxyz",
	}
	if len(Num_str)==0:
		print(str(res_ls))
	else:
		res_ls=[n for n in num_char_map[Num_str[0]]]

		for number in Num_str[1:]:
			res_ls=[m+n for m in res_ls for n in num_char_map[number]]
	
	print(str(res_ls))	

TeleNum_Comb("95")

