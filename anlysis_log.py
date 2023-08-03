
import re

def output_sample(filename:str,outputname:str):
    ret=[]
    with open(filename,'r') as f:
        content = f.read()
# INFO:2023-05-30 20:51:58,168:seleced clients:(2, 3, 5, 6)
        p = r'\([\s]*\d[\s]*,[\s]*\d[\s]*,[\s]*\d[\s]*,[\s]*\d[\s]*\)'
        ret = re.findall(p,content)
    
    out = open(outputname,'w')
    for line in ret:
        out.write(line+'\n')
    out.close()
 
 
            
if __name__ == '__main__':
    print('dd')
    output_sample(r'logs\90\output.log',r'.\log-temp-result.txt')