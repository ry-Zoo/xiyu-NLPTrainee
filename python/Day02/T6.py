def longestPalindrome(raw_str):
	lennum=len(raw_str)
	if lennum == 0:
		print("empty str")
		exit(0)
	else:
		for i in range(0,lennum):
			#print(str(raw_str[i])+'\n')
			for c in range(i+1,lennum):
				#print(str(raw_str[c])+'?')
				if raw_str[i]==raw_str[c]:
					test_str=raw_str[i:c+1]
					ls.append(test_str)
				else:
					continue

def is_Palindrome(ls_item):
	if ls_item[::-1]==ls_item:
		new_ls.append(ls_item)



	
s=''
ls=[]
longestPalindrome(s)
#print(ls)
new_ls=[]
for item in ls:
	is_Palindrome(item)
#print(new_ls)
new_ls.sort(key=lambda x:len(x),reverse=True)
maxPalindrome=ls[0]
print(str(maxPalindrome))


