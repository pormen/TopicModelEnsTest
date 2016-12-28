#!/usr/bin/env python
# -*- coding: utf-8 -*-


print "1: crear; 2: muestra"
seleccion = int(raw_input("Seleccione: "))



if seleccion == 1:
    print "1: train; 2: test"
    dm = int(raw_input("Seleccione datamode: "))
    if dm == 1:
        dataMode = 'train';
    if dm == 2:
        dataMode = 'test';
    print(__doc__)

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    from sklearn.cluster import MiniBatchKMeans, KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.cm as cm

    from sklearn.datasets import fetch_20newsgroups
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    # Load some categories from the training set
    #categories = [    'alt.atheism']
    # Uncomment the following to do the analysis on all the categories
    categories = None

    print("Loading 20 newsgroups dataset for categories:"); print(categories)



    newsgroups_train = fetch_20newsgroups(subset=dataMode,remove=('headers', 'footers', 'quotes'),categories=categories)

    print len(list(newsgroups_train.filenames)); print len(list(newsgroups_train.data))

    fileName = ""; folderName = ""; corpus = "";

    import io

    f2= open("C:/Mallet/20newsFormated/"+dataMode+"/corpus.txt","w")
    f3= open("C:/Mallet/20newsFormated/"+dataMode+"/metadata.txt","w")

    f_alt_atheism = open("C:/Mallet/20newsFormated/"+dataMode+"/alt_atheism.txt","w")
    f_comp_graphics = open("C:/Mallet/20newsFormated/"+dataMode+"/comp_graphics.txt","w")
    f_comp_os_ms_windows_misc= open("C:/Mallet/20newsFormated/"+dataMode+"/comp_os_ms_windows_misc.txt","w")
    f_comp_sys_ibm_pc_hardware= open("C:/Mallet/20newsFormated/"+dataMode+"/comp_sys_ibm_pc_hardware.txt","w")
    f_comp_sys_mac_hardware= open("C:/Mallet/20newsFormated/"+dataMode+"/comp_sys_mac_hardware.txt","w")

    f_comp_windows_x= open("C:/Mallet/20newsFormated/"+dataMode+"/comp_windows_x.txt","w")
    f_misc_forsale= open("C:/Mallet/20newsFormated/"+dataMode+"/misc_forsale.txt","w")
    f_rec_autos= open("C:/Mallet/20newsFormated/"+dataMode+"/rec_autos.txt","w")
    f_rec_motorcycles= open("C:/Mallet/20newsFormated/"+dataMode+"/rec_motorcycles.txt","w")
    f_rec_sport_baseball= open("C:/Mallet/20newsFormated/"+dataMode+"/rec_sport_baseball.txt","w")

    f_rec_sport_hockey= open("C:/Mallet/20newsFormated/"+dataMode+"/rec_sport_hockey.txt","w")
    f_sci_crypt= open("C:/Mallet/20newsFormated/"+dataMode+"/sci_crypt.txt","w")
    f_sci_electronics= open("C:/Mallet/20newsFormated/"+dataMode+"/sci_electronics.txt","w")
    f_sci_med= open("C:/Mallet/20newsFormated/"+dataMode+"/sci_med.txt","w")
    f_sci_space= open("C:/Mallet/20newsFormated/"+dataMode+"/sci_space.txt","w")

    f_soc_religion_christian= open("C:/Mallet/20newsFormated/"+dataMode+"/soc_religion_christian.txt","w")
    f_talk_politics_guns= open("C:/Mallet/20newsFormated/"+dataMode+"/talk_politics_guns.txt","w")
    f_talk_politics_mideast= open("C:/Mallet/20newsFormated/"+dataMode+"/talk_politics_mideast.txt","w")
    f_talk_politics_misc= open("C:/Mallet/20newsFormated/"+dataMode+"/talk_politics_misc.txt","w")
    f_talk_religion_misc= open("C:/Mallet/20newsFormated/"+dataMode+"/talk_religion_misc.txt","w")



    for l1, l2 in zip(list(newsgroups_train.filenames),list(newsgroups_train.data)):
                listDir = str(l1).split("\\");
                fileName =  listDir.pop();
                folderName =  listDir.pop();
                #print folderName
                #print fileName

                string = "";
                newDoc = [];
                for word in l2.split(" "):
                    import unicodedata

                    if isinstance(word, unicode):
                        word = unicodedata.normalize('NFKD', word).encode('ascii','ignore')
                    if isinstance(word, str):
                        word = str(word).replace('\n',' ').replace('\t',' ').encode('utf-8')
                    if len(str(word)) > 3:

                        stop2 = ["a’s", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards",
                                 "again", "against", "ain’t", "all", "allow", "allows", "almost", "alone", "along", "already", "also",
                                 "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow",
                                 "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate",
                                 "are", "aren’t", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away",
                                 "awfully", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
                                 "behind", "being", "believe", "below", "beside","besides", "best", "better", "between", "beyond", "both",
                                 "brief", "but", "by", "c’mon", "c’s", "came", "can", "can’t", "cannot", "cant", "cause", "causes", "certain",
                                 "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider",
                                 "considering", "contain", "containing", "contains", "corresponding", "could", "couldn’t", "course", "currently",
                                 "definitely", "described", "despite", "did", "didn’t", "different", "do", "does", "doesn’t", "doing", "don’t",
                                 "done", "down", "downwards", "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough",
                                 "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere",
                                 "ex", "exactly", "example", "except", "far", "few", "fifth", "first", "five", "followed", "following", "follows",
                                 "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "get", "gets", "getting", "given",
                                 "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had", "hadn’t", "happens", "hardly", "has",
                                 "hasn’t", "have", "haven’t", "having", "he", "he’s", "hello", "help", "hence", "her", "here", "here’s", "hereafter",
                                 "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how",
                                 "howbeit", "however", "i’d", "i’ll", "i’m", "i’ve", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc",
                                 "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn’t", "it",
                                 "it’d", "it’ll", "it’s", "its", "itself", "just", "keep", "keeps", "kept", "know", "knows", "known", "last", "lately",
                                 "later", "latter", "latterly", "least", "less", "lest", "let", "let’s", "like", "liked", "likely", "little", "look",
                                 "looking", "looks", "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more",
                                 "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary",
                                 "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone",
                                 "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often", "oh", "ok", "okay",
                                 "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours",
                                 "ourselves", "out", "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed",
                                 "please", "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather", "rd", "re",
                                 "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "said", "same",
                                 "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
                                 "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn’t",
                                 "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes",
                                 "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup",
                                 "sure", "t’s", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that’s", "thats",
                                 "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there’s", "thereafter", "thereby",
                                 "therefore", "therein", "theres", "thereupon", "these", "they", "they’d", "they’ll", "they’re", "they’ve", "think",
                                 "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus",
                                 "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two",
                                 "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful",
                                 "uses", "using", "usually", "value", "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasn’t", "way",
                                 "we", "we’d", "we’ll", "we’re", "we’ve", "welcome", "well", "went", "were", "weren’t", "what", "what’s", "whatever",
                                 "when", "whence", "whenever", "where", "where’s", "whereafter", "whereas", "whereby", "wherein", "whereupon",
                                 "wherever", "whether", "which", "while", "whither", "who", "who’s", "whoever", "whole", "whom", "whose", "why",
                                 "will", "willing", "wish", "with", "within", "without", "won’t", "wonder", "would", "would", "wouldn’t", "yes",
                                 "yet", "you", "you’d", "you’ll", "you’re", "you’ve", "your", "yours", "yourself", "yourselves", "zero"]

                        if word not in stop2:

                            word = str(word).replace('{',' ').replace('}',' ').replace('(',' ').replace(')',' ')
                            word = str(word).replace('[',' ').replace(']',' ').replace('.',' ').replace(',',' ')
                            word = str(word).replace(':',' ').replace(';',' ').replace('+',' ').replace('-',' ')
                            word = str(word).replace('*',' ').replace('/',' ').replace('&',' ').replace('|',' ')
                            word = str(word).replace('<',' ').replace('>',' ').replace('=',' ').replace('#',' ')
                            word = str(word).replace('?',' ').replace('!',' ').replace('$',' ').replace('\\',' ')
                            word = str(word).replace('_',' ').replace('^',' ').replace("'s",' ').replace('/',' ')
                            word = str(word).replace('"',' ').replace("'",' ')

                            nouns = ['i','you','he','she','it','we','they','me','you','his','her','it','us','them',
                                     'my','your','his','her','its','our','your','their','mine','yours','his','hers','ours','yours','theirs',
                                     'myself','yourself','himself','herself','itself','ourselves','yourselves','themselves']

                            stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]




                            if not("'" in word):
                                if not(any(char.isdigit() for char in word)):
                                    if len(word) > 3:
                                        if not('@' in word):
                                            if not(' ' in word):
                                                word = word.lower()
                                                if word not in nouns:
                                                    if word not in stopwords:
                                                        word = lmtzr.lemmatize(word)
                                                        newDoc.append(word);
                            string = string + " " + word


                if len(newDoc) < 15:
                    continue

                newDoc = " ".join(newDoc);



                if folderName == "alt.atheism": f_alt_atheism.write("D T: "+str(newDoc)+"\n")
                if folderName == "comp.graphics": f_comp_graphics.write("D T: "+str(newDoc)+"\n")
                if folderName == "comp.os.ms-windows.misc": f_comp_os_ms_windows_misc.write("D T: "+str(newDoc)+"\n")
                if folderName == "comp.sys.ibm.pc.hardware": f_comp_sys_ibm_pc_hardware.write("D T: "+str(newDoc)+"\n")
                if folderName == "comp.sys.mac.hardware": f_comp_sys_mac_hardware.write("D T: "+str(newDoc)+"\n")
                if folderName == "comp.windows.x": f_comp_windows_x.write("D T: "+str(newDoc)+"\n")
                if folderName == "misc.forsale": f_misc_forsale.write("D T: "+str(newDoc)+"\n")
                if folderName == "rec.autos": f_rec_autos.write("D T: "+str(newDoc)+"\n")
                if folderName == "rec.motorcycles": f_rec_motorcycles.write("D T: "+str(newDoc)+"\n")
                if folderName == "rec.sport.baseball": f_rec_sport_baseball.write("D T: "+str(newDoc)+"\n")
                if folderName == "rec.sport.hockey": f_rec_sport_hockey.write("D T: "+str(newDoc)+"\n")
                if folderName == "sci.crypt": f_sci_crypt.write("D T: "+str(newDoc)+"\n")
                if folderName == "sci.electronics": f_sci_electronics.write("D T: "+str(newDoc)+"\n")
                if folderName == "sci.med": f_sci_med.write("D T: "+str(newDoc)+"\n")
                if folderName == "sci.space": f_sci_space.write("D T: "+str(newDoc)+"\n")
                if folderName == "soc.religion.christian": f_soc_religion_christian.write("D T: "+str(newDoc)+"\n")
                if folderName == "talk.politics.guns": f_talk_politics_guns.write("D T: "+str(newDoc)+"\n")
                if folderName == "talk.politics.mideast": f_talk_politics_mideast.write("D T: "+str(newDoc)+"\n")
                if folderName == "talk.politics.misc": f_talk_politics_misc.write("D T: "+str(newDoc)+"\n")
                if folderName == "talk.religion.misc": f_talk_religion_misc.write("D T: "+str(newDoc)+"\n")



                f3.write(fileName+folderName+"\n")
                f2.write("D T: "+str(newDoc)+"\n")

                    #import sys; sys.exit("Error message")

    f2.close()
    f3.close()

    f_alt_atheism.close(); f_comp_graphics.close(); f_comp_os_ms_windows_misc.close();f_comp_sys_ibm_pc_hardware.close();f_comp_sys_mac_hardware.close()
    f_comp_windows_x.close(); f_misc_forsale.close(); f_rec_autos.close(); f_rec_motorcycles.close(); f_rec_sport_baseball.close()
    f_rec_sport_hockey.close();f_sci_crypt.close(); f_sci_electronics.close(); f_sci_med.close(); f_sci_space.close()
    f_soc_religion_christian.close(); f_talk_politics_guns.close(); f_talk_politics_mideast.close() ;f_talk_politics_misc.close(); f_talk_religion_misc.close()


if seleccion == 2:
    import random

    print "1: train; 2: test"
    dm = int(raw_input("Seleccione datamode: "))
    if dm == 1:
        dataMode = 'train';
    if dm == 2:
        dataMode = 'test';
    limite = 0;

    folderList = ["/corpus.txt","/alt_atheism.txt", "/comp_graphics.txt","/comp_os_ms_windows_misc.txt",
                  "/comp_sys_ibm_pc_hardware.txt","/comp_sys_mac_hardware.txt","/comp_windows_x.txt", "/misc_forsale.txt",
                  "/rec_autos.txt","/rec_motorcycles.txt","/rec_sport_baseball.txt", "/rec_sport_hockey.txt",
                  "/sci_crypt.txt","/sci_electronics.txt","/sci_med.txt","/sci_space.txt",
                  "/soc_religion_christian.txt","/talk_politics_guns.txt","/talk_politics_mideast.txt","/talk_politics_misc.txt",
                  "/talk_religion_misc.txt"]
    SAMPLEDfolderList = ["/corpus_SAMPLED.txt","/alt_atheism_SAMPLED.txt", "/comp_graphics_SAMPLED.txt","/comp_os_ms_windows_misc_SAMPLED.txt",
                         "/comp_sys_ibm_pc_hardware_SAMPLED.txt","/comp_sys_mac_hardware_SAMPLED.txt","/comp_windows_x_SAMPLED.txt", "/misc_forsale_SAMPLED.txt",
                         "/rec_autos_SAMPLED.txt","/rec_motorcycles_SAMPLED.txt","/rec_sport_baseball_SAMPLED.txt", "/rec_sport_hockey_SAMPLED.txt",
                         "/sci_crypt_SAMPLED.txt","/sci_electronics_SAMPLED.txt","/sci_med_SAMPLED.txt","/sci_space_SAMPLED.txt",
                         "/soc_religion_christian_SAMPLED.txt","/talk_politics_guns_SAMPLED.txt","/talk_politics_mideast_SAMPLED.txt","/talk_politics_misc_SAMPLED.txt",
                         "/talk_religion_misc_SAMPLED.txt"]

    folderSize = [];

    folderSample = []

    for file in folderList:
        f = open("C:/Mallet/20newsFormated/"+dataMode+file,"r");
        cont = 0;
        for elem in f:
            cont = cont + 1
        folderSize.append(cont)

    print folderSize

    for num in folderSize:
        folderNum = int(num*0.05);
        folderSample.append(random.sample(range(num), folderNum))

    print folderSample



    f3 = open("C:/Mallet/20newsFormated/"+dataMode+"/AllSampled.txt","w");
    for file, sampleList, num, sampleFile in zip(folderList, folderSample, folderSize, SAMPLEDfolderList):
        f = open("C:/Mallet/20newsFormated/"+dataMode+file,"r");
        f2 = open("C:/Mallet/20newsFormated/"+dataMode+sampleFile,"w");


        newSampleList =  sorted(sampleList)
        print newSampleList

        for n, elem in zip(range(num), f):


            if n+1 in newSampleList:
                if "D T:" in elem:
                    print n+1, elem
                    f2.write(elem)
                    if file != "/corpus.txt":
                        f3.write(elem)

        f2.close()

    f3.close()
        #import sys; sys.exit("Error message")


#"C:/Mallet/20newsFormated/"+dataMode




