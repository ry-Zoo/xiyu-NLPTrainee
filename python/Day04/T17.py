def ZshapeConvert(raw_str,row):
	length=len(raw_str)
	'''
	if length%(2(row-1))==0:
		column=length//(2(row-1))*(row-1)
	else:
		column=length//(2(row-1))*(row-1)+(length%(2(row-1))-row)+1
	'''
	area=[["" for i in range(0,row)]for j in range(0,column]
	if row==1:
		return raw_str
	else :        
		i=0 
		while i<length:
			for j in range(row):
				if i<length:
					area[j]+=raw_str[i]
					i+=1
			for j in range(row-2,0,-1):
				if i<length:
					area[j]+=raw_str[i]
					i+=1
		return "".join(area)


ZshapeConvert("0123456789ABCDEF")


				
