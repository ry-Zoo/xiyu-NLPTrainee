def get_sub_string(s):
	substring=[]
	length=len(s)
	for i in range(length):
		for j in range(length-i):
			substring.append(s[i:i+j+1])
	return substring

def is_palindromel(subs):
	if subs[::-1]==subs:
		valid_substring.append(subs)
		print (str(subs))

s="jhhhkkmaba"
substring = get_sub_string(s)
#print(substring)
valid_substring =[]
for item in substring :
	is_palindromel(item)
	
#print(valid_substring)



