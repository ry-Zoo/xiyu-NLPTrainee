from collections import Counter

def canConstruct(ransom,magazine):
	dict_ransom=Counter(ransom)
	dict_magazine=Counter(magazine)
	count = 0
	for k in dict_magazine.keys():
		if k in dict_ransom.keys(): 
			if dict_ransom[k]<=dict_magazine[k]:
				count+=1
	if count==len(dict_ransom):
		print("True")
	else:
		print("False")


canConstruct("aat","aab")