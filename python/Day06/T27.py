import re

def function_atoi(raw_str):
	res_num=0
	real_str=raw_str.strip()
	if real_str[0]=="-":
		pattern=r"\d+"
		n_str=re.findall(pattern,real_str[1:])[::-1]
		for s in range(1,len(n_str[-1])+1):
			res_num=res_num-num_dic[n_str[-1][s]]*s
	elif real_str[0]=="+":
		pattern=r"\d+"
		n_str=re.findall(pattern,real_str[1:])[::-1]
		for s in range(1,len(n_str[-1])+1):
			res_num=res_num+num_dic[n_str[-1][s]]*s
	elif real_str[0] in num_dic.keys():
		pattern=r"\d+"
		n_str=re.findall(pattern,real_str)[::-1]
		print(n_str[-1])
		for s in range(0,len(n_str[-1])):
			res_num=res_num+num_dic[n_str[-1][s]]*s
	else:
		res_num = 0
	if res_num>(2**31) or res_num<(-2**31):
		return  0
	else:
		return res_num

num_dic={
	"1":1,
	"2":2,
	"3":3,
	"4":4,
	"5":5,
	"6":6,
	"7":7,
	"8":8,
	"9":9,
	"0":0
}

function_atoi("   123.22ddd  ")


'''
class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', s.lstrip())), 2**31 - 1), -2**31)

作者：QQqun902025048
链接：https://leetcode-cn.com/problems/two-sum/solution/python-1xing-zheng-ze-biao-da-shi-by-knifezhu/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

'''