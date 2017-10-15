path = "/home/mohamed/Desktop/tdt2_em_v4_0/doc/tdt2topics.html"

start_read = False

found_event = False
topic_index = 1
with open(path,'r') as file:
    expl = ""
    for line in file:
        if "TOPIC EXPLICATION" in line and ("</LI>" not in line):
            start_read = True
        elif "TOPIC EXPLICATION" in line and ("</LI>" in line):
            found_event = True

        if found_event and "<BR>" in line:
            start_read = True
            found_event = False
        if start_read:
            expl += line.strip().replace("</LI>", "").replace("TOPIC EXPLICATION:", "")\
                .replace("<BR>","").replace("</UL>","")\
                .replace("<U>","").replace("</U>","") + " "

        if start_read and ("</LI>" in line or "</UL>" in line):
            start_read = False
            print str(topic_index) + " ",
            print expl,
            print "\n",
            expl = ""
            topic_index += 1