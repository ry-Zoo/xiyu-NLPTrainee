def Valid_iP(ip_str):
	if ":" in  ip_str:
		ipv6_ls=ip_str.split(':')
		flag=False
		if len(ipv6_ls)!=8:
			flag=False
		if "" in ipv6_ls:
			flag=False
		else:
			for m in ipv6_ls:
				if len(m) > 4:
					flag=False
					break
		
		if flag==True:
			print("IPv6")	
		else:
			print("Neither")
	else:
		ipv4_ls=ip_str.split('.')
		flag=False
		if len(ipv4_ls)!=4:
			flag=False
		else:
			for n in ipv4_ls: 
				if len(n) >= 2 and n[0]=="0":
					continue
				if int(n)>255 or int(n)<0:
					continue
				else:
					flag=True
		if flag==True:
			print("IPv4")	
		else:
			print("Neither")

#Valid_iP("256.256.256.256")
Valid_iP("2001:0db8:256e:0256f:8896:3333:1211:6669")