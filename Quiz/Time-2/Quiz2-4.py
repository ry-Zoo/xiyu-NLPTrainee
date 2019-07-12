class Solution:
    def myAtoi(self, raw_str: str) -> int:
        raw_str=raw_str.strip()
        result=0
        try:
            res=re.search(r'^[\+|\-]?\d+',raw_str).group()
            if int(res)>(2**31) : 
                return 2**31
            elif int(res)<((-2)**31):
                return ((-2)**31)
            else:
                result=int(res)
        except:
            return 0

        return result