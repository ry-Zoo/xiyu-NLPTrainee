def repeateStr(raw_str):
	r_length=len(raw_str)
	#print(raw_str)
	check_flag=False
	for w in range(1,r_length//2+1):
		if r_length%w==0:
			pattern=raw_str[:w]
			t=w
			while t<r_length and raw_str[t:t+w]==pattern:
				t+=w
			if t==r_length:
				check_flag=True
	if check_flag==True:
		print('True')
	else:
		print('False')

raw_str=input()

repeateStr(raw_str)

