
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        # ������ȱ���
        def dfs(results,result,s,level):
            # print(results,s)
            if len(s) == 0 and level == 4:
                results.add(".".join(result))
                return None
            if level >= 4: # ����4��ip�Ļ�������
                return None
            for i in range(1,4):# ÿ��ip��ַ�����λ����
                if s[:i] != "" and 0 <= int(s[:i]) <= 255: # ��֤ÿ��ip��Ϊ�� ����0-255
                    if len(s[:i]) >= 2 and s[0] == "0":continue # ��֤ÿ��ipû�� 01.011.010������
                    dfs(results,result + [s[:i]],s[i:],level + 1)
        results = set()
        dfs(results,[],s,0)
        return list(results)