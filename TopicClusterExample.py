

import os
import random
import numpy
import math

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import numpy

import numpy as np
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score
import time
#import numpy as np
#import matplotlib.pyplot as plt
import random
from random import shuffle

def testCorpusPerplexity(trainWords, testWords, testDocs, matrix_Z_Words_Total):
    print "===INICIO testCorpusPerplexity==="
    import numpy as np
    import math

    rows = matrix_Z_Words_Total.shape[0]
    cols = matrix_Z_Words_Total.shape[1]

    #print "rows", rows,float(1/rows)
    topicPriors = float(1/float(rows))
    #print "topicPriors", topicPriors
    print testWords

    #print matrix_Z_Words_Total

    aux_Z_Words_Total = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            aux_Z_Words_Total[i,j] = matrix_Z_Words_Total.item(i,j) + 0.0000000000000001

    aux_Z_Words_Total = aux_Z_Words_Total/numpy.array([aux_Z_Words_Total.sum(axis=1)]).transpose()


    #print aux_Z_Words_Total


    testWordsIndex = [];
    for elem in testWords:
        cont = 0
        for elem2 in trainWords:
            if elem == elem2:
                testWordsIndex.append(cont)
                break;
            else:
                cont = cont+ 1;

    #print testWordsIndex
    mult= 1
    suma = 0.0;
    sumaPerp= 0;

    for indx in testWordsIndex:
        for row in range(rows):
            if float(aux_Z_Words_Total.item(row,indx)) != 0:
                sumaPerp = math.log(float(aux_Z_Words_Total.item(row,indx))*topicPriors) + sumaPerp


    print sumaPerp

    #print aux_Z_Words_Total
    #print trainWords
    #print testWords
    #print testDocs
    print "===FIN testCorpusPerplexity==="


def formatingFilesTest(numTopic,file,folder,alpha, beta):

    f2= open("C:/Mallet/testCMD/"+folder+"/WordPerDoc-Test.txt","w")
    f3= open("C:/Mallet/testCMD/"+folder+"/LDAoutput-Test.txt","w")


    os.chdir( 'c:\\Mallet\\bin' )
    cmd1 = "C:\\Mallet\\bin\\mallet import-file --input C:\\Mallet\\testCMD\\"+file +" ";
    cmd1 = cmd1 + "--output C:\\Mallet\\testCMD\\"+folder+"\\training2.mallet ";
    cmd1 = cmd1 + "--keep-sequence --remove-stopwords --print-output ";


    num_top_words = 5;

    cmd2 = "C:\\Mallet\\bin\\mallet train-topics --input C:\\Mallet\\testCMD\\"+folder+"\\training2.mallet ";
    cmd2 = cmd2 + "--output-state C:\\Mallet\\testCMD\\"+folder+"\\topic-state-Test.gz --output-topic-keys C:\\Mallet\\testCMD\\"+folder+"\\tutorial_keys-Test.txt ";
    cmd2 = cmd2 + "--output-doc-topics C:\\Mallet\\testCMD\\"+folder+"\\doc_topic-Test.txt --topic-word-weights-file C:\\Mallet\\testCMD\\"+folder+"\\top_words_weights-Test.txt ";
    cmd2 = cmd2 + "--word-topic-counts-file C:\\Mallet\\testCMD\\"+folder+"\\words_count_file-Test.txt "
    cmd2 = cmd2 + "--num-topics " + str(numTopic) + " --num-top-words " + str(num_top_words) + " ";
    cmd2 = cmd2 + "--evaluator-filename C:\\Mallet\\testCMD\\"+folder+"\\eval-Test.txt --alpha "+str(alpha)+" --beta "+str(beta)+"";


    cmd3 = "C:\\Mallet\\bin\\mallet evaluate-topics --evaluator C:\\Mallet\\testCMD\\"+folder+"\\eval-Test.txt ";
    cmd3 = cmd3 + "--input C:\\Mallet\\testCMD\\"+folder+"\\training2.mallet "
    cmd3 = cmd3 + "--output-doc-probs C:\\Mallet\\testCMD\\"+folder+"\\docprobs-Test.dat --output-prob C:\\Mallet\\testCMD\\"+folder+"\\prob-Test.dat";

    cmd4 = "C:\\Mallet\\bin\\mallet run cc.mallet.util.DocumentLengths --input C:\\Mallet\\testCMD\\"+folder+"\\training2.mallet";

    f1= open("C:/Mallet/testCMD/"+folder+"/WordPerDoc-Test.txt","w")
    f2= open("C:/Mallet/testCMD/"+folder+"/LDAoutput-Test.txt","w")
    f3= open("C:/Mallet/testCMD/"+folder+"/EvalTopic-Test.txt","w")
    f4= open("C:/Mallet/testCMD/"+folder+"/DocsLength-Test.txt","w")

    dir = os.popen(cmd1).readlines()
    dir2 = os.popen(cmd2).readlines()
    dir3 = os.popen(cmd3).readlines()
    dir4 = os.popen(cmd4).readlines()

    for lines in dir:
        f1.write(lines);
    for lines in dir2:
        f2.write(lines);
    for lines in dir3:
        f3.write(lines);
    for lines in dir4:
        f4.write(lines);

    f1.close()
    f2.close()
    f3.close()
    f4.close()

def formatingFiles(numTopic,file,folder,alpha, beta):



    f2= open("C:/Mallet/testCMD/"+folder+"/WordPerDoc.txt","w")
    f3= open("C:/Mallet/testCMD/"+folder+"/LDAoutput.txt","w")


    os.chdir( 'c:\\Mallet\\bin' )
    cmd1 = "C:\\Mallet\\bin\\mallet import-file --input C:\\Mallet\\testCMD\\"+file +" ";
    cmd1 = cmd1 + "--output C:\\Mallet\\testCMD\\"+folder+"\\training.mallet ";
    cmd1 = cmd1 + "--keep-sequence --remove-stopwords --print-output ";

    num_top_words = 5;

    cmd2 = "C:\\Mallet\\bin\\mallet train-topics --input C:\\Mallet\\testCMD\\"+folder+"\\training.mallet ";
    cmd2 = cmd2 + "--output-state C:\\Mallet\\testCMD\\"+folder+"\\topic-state.gz --output-topic-keys C:\\Mallet\\testCMD\\"+folder+"\\tutorial_keys.txt ";
    cmd2 = cmd2 + "--output-doc-topics C:\\Mallet\\testCMD\\"+folder+"\\doc_topic.txt --topic-word-weights-file C:\\Mallet\\testCMD\\"+folder+"\\top_words_weights.txt ";
    cmd2 = cmd2 + "--word-topic-counts-file C:\\Mallet\\testCMD\\"+folder+"\\words_count_file.txt "
    cmd2 = cmd2 + "--num-topics " + str(numTopic) + " --num-top-words " + str(num_top_words) + " ";
    cmd2 = cmd2 + "--evaluator-filename C:\\Mallet\\testCMD\\"+folder+"\\eval.txt --alpha "+str(alpha)+" --beta "+str(beta)+"";

    cmd3 = "C:\\Mallet\\bin\\mallet evaluate-topics --evaluator C:\\Mallet\\testCMD\\"+folder+"\\eval.txt ";
    cmd3 = cmd3 + "--input C:\\Mallet\\testCMD\\"+folder+"\\training.mallet "
    cmd3 = cmd3 + "--output-doc-probs C:\\Mallet\\testCMD\\"+folder+"\\docprobs.dat --output-prob C:\\Mallet\\testCMD\\"+folder+"\\prob.dat";

    cmd4 = "C:\\Mallet\\bin\\mallet run cc.mallet.util.DocumentLengths --input C:\\Mallet\\testCMD\\"+folder+"\\training.mallet";

    f1= open("C:/Mallet/testCMD/"+folder+"/WordPerDoc.txt","w")
    f2= open("C:/Mallet/testCMD/"+folder+"/LDAoutput.txt","w")
    f3= open("C:/Mallet/testCMD/"+folder+"/EvalTopic.txt","w")
    f4= open("C:/Mallet/testCMD/"+folder+"/DocsLength.txt","w")

    dir = os.popen(cmd1).readlines()
    dir2 = os.popen(cmd2).readlines()
    dir3 = os.popen(cmd3).readlines()
    dir4 = os.popen(cmd4).readlines()

    for lines in dir:
        f1.write(lines);
    for lines in dir2:
        f2.write(lines);
    for lines in dir3:
        f3.write(lines);
    for lines in dir4:
        f4.write(lines);

    f1.close()
    f2.close()
    f3.close()
    f4.close()

def LDAPerplexity(numTopic,file,folder):
    print "===INICIO Funcion LDAPerplexity==="

    #print "TOPICOS " + str(numTopic)

    import numpy as np
    import math
    f = open("C:/Mallet/testCMD/"+folder+"/words_count_file.txt","r")
    f2 = open("C:/Mallet/testCMD/"+folder+"/doc_topic.txt","r")
    f3 = open("C:/Mallet/testCMD/"+folder+"/WordPerDoc.txt","r")

    #print folder
    #print "C:/Mallet/testCMD/"+folder+"/words_count_file.txt"

    numTerm = 0;
    for line in f:
        numTerm  = numTerm + 1;

    #print numTerm
    f.close();
    numDocs = 0;
    for line in f2:
        if "#doc name" in line:
            continue;
        numDocs  = numDocs + 1;



    priorsDoc = [];
    for x in range(0, numDocs):
        priorsDoc.append(float(1/float(numDocs)));

    f2.close()



    #=======FIN DATOS DOC

    #=======INICIO DOC TERMS

    print numDocs, numTerm
    matrixDocTerms = np.zeros((int(numDocs),int(numTerm)))

    wordList = [];
    docNumber = 0;
    for line in f3:
        if "D1" in line:
            continue;
        if "target: T" in line:
            continue;
        if "name: D" in line:
            docNumber = docNumber + 1;
            continue;
        if len(line) == 1:
            continue;
        if "input: 0" in line:
            wordList.append(str(docNumber)+" " + line.split(" ")[2] + " " + line.split(" ")[3].replace("(","").replace(")",""));
        else:
            wordList.append(str(docNumber)+" " + line.split(" ")[1] + " " + line.split(" ")[2].replace("(","").replace(")",""));

    print "wordList"
    #print wordList



    for elem in wordList:
        if int(str(elem).split(" ")[0]) < numDocs:
            matrixDocTerms[int(str(elem).split(" ")[0])][int(str(elem).split(" ")[2])] = 1



    print "matrixDocTerms"
    #print matrixDocTerms


    f2 = open("C:/Mallet/testCMD/"+folder+"/doc_topic.txt","r")
    matrixDocTop = np.zeros((int(numDocs),int(numTopic)))

    for line in f2:
        if "#doc name topic proportion" in line:
            continue;
        else:
            cont = 1;
            docAux =str(line).replace("\t\n","").split("\t")
            for elem in docAux:
                if cont < len(docAux):
                    if cont % 2 == 0:
                        matrixDocTop[int(docAux[0])][int(docAux[cont])] = float(docAux[cont + 1]);

                else: break;
                cont = cont + 1;

    #print "Doc Top"
    #print matrixDocTop

    #=======FIN DOC TOPIC

    #=======INICIO TOPIC TERMS


    vectTopCorpus =  matrixDocTop.sum(axis=0)/matrixDocTop.sum(axis=0).sum();
    #print vectTopCorpus

    f2.close();

    f = open("C:/Mallet/testCMD/"+folder+"/words_count_file.txt","r")
    matrixTopWord = np.zeros((int(numTopic),int(numTerm)))

    wordTopDist = []
    wordId = []
    for line in f:
        wordAux = str(line).replace("\n","").split(" ");
        wordId.append(str(line).replace("\n","").split(" ")[0]+":"+str(line).replace("\n","").split(" ")[1])

        cont  = 0;
        weightList = []
        sum = 0;
        for elem in wordAux:
            if cont > 1:
                weightList.append(int(elem.split(":")[1]))
                sum = sum + int(elem.split(":")[1]) ;

            cont = cont +1;
        weAux = [];
        weAux.append("#");
        weAux.append("#");

        for num in weightList:
            weAux.append(float(float(num)/float(sum)));

        weightList = weAux;

        cont = 0;
        for elem in wordAux:
            if cont > 1:
                wordTopDist.append(str(wordAux[0])+":"+str(elem.split(":")[0])+":"+str(weightList[cont]));
            cont = cont + 1;

    f.close();

    for elem in wordTopDist:
        matrixTopWord[int(elem.split(":")[1])][int(elem.split(":")[0])] = float(elem.split(":")[2])

    print "matrixTopWord"


    import numpy
    from collections import Counter;

    docs  = [];
    doc = [];

    priorsWordsPerDoc = [];

    priorsCorpus = []

    file = open("C:/Mallet/testCMD/"+folder+"/WordperDoc.txt", "r");

    start = 0;
    for line in file:
        if "name: D" in line: continue;
        if "target: T:" in line: continue;
        if "input:" in line:
           line = line.replace("input: ","").replace("\n","")
           doc.append(line.split(" "))
           continue;
        line = line.replace("\n","")

        if len(line) != 0:
            doc.append(line.split(" "))

        if len(line) == 0:
            listaAux = []
            listDocs = []
            for elem in doc:
                listDocs =elem[1:];
                listaAux.append(" ".join(str(x) for x in listDocs))
                priorsCorpus.append(" ".join(str(x) for x in listDocs))

            doc = listaAux

            priorsWords = [];
            suma = 0;
            priorsWords =  Counter(doc).items()

            for elem in priorsWords:
                suma = suma +  elem[1];

            listaAux =[];
            lista = [];
            for elem in priorsWords:
                lista = list(elem)
                lista.append(float(elem[1])/float(suma));
                listaAux.append(lista)

            priorsWords =  listaAux;

            priorsWordsPerDoc.append(priorsWords);
            doc = [];

    file.close();

    #print priorsWordsPerDoc

    suma = 0;
    for elem in Counter(priorsCorpus).items():
        suma = suma +  elem[1];


    listaAux =[];
    lista = [];
    for elem in Counter(priorsCorpus).items():
        lista = list(elem)
        lista.append(float(elem[1])/float(suma));
        listaAux.append(lista)

    priorsCorpus =  listaAux;





    matrixProbDocTerms = matrixDocTerms;
    indexDoc = 0
    for elem in priorsWordsPerDoc:
        for elem2 in elem:
            indexTerm =  str(elem2[0]).split(" ")[1].replace("(","").replace(")","")
            matrixProbDocTerms[int(indexDoc)][int(indexTerm)]= float("{0:.4f}".format(elem2[2]))
        indexDoc = indexDoc + 1


    matrixPriorsCorpus = np.zeros((int(1),int(numTerm)))



    for elem in priorsCorpus:
        indexTerm =  str(elem[0]).split(" ")[1].replace("(","").replace(")","")
        matrixPriorsCorpus[int(0)][int(indexTerm)]= float("{0:.8f}".format(elem[2]))


    print "matrixDocTerms"
    #print matrixDocTerms

    #print matrixDocTerms

    #print matrixDocTop

    #print matrixTopWord

    #print np.dot(matrixTopWord.transpose(), )
    #print matrixProbDocTerms

    #print matrixPriorsCorpus


    #print priorsDoc
    #print ""
    #print vectTopCorpus
    #print ""

    matAux = numpy.zeros((int(len(vectTopCorpus)),int(len(priorsDoc))))
    x = 0;
    y = 0;

    for x1 in vectTopCorpus:#P(META)
        for y1 in priorsDoc:#P(DOC)
            matAux[x][y] =  float(y1)/float(x1)
            y = y + 1;
        x = x + 1;
        y = 0;

    #print matAux
    #print ""
    #print matrixDocTop.transpose()
    #print ""
    #print matrixDocTop.transpose()*matAux

    matrixDocTermFinal =  np.dot(matrixTopWord.transpose(),matrixDocTop.transpose()*matAux)

    #print matrixPriorsCorpus[0]
    #print priorsDoc

    matAux = numpy.zeros((int(len(matrixPriorsCorpus[0])),int(len(priorsDoc))))
    x = 0;
    y = 0;

    print "matrixPriorsCorpus[0]"
    #print matrixPriorsCorpus[0]

    for x1 in matrixPriorsCorpus[0]:#P(META)
        for y1 in priorsDoc:#P(DOC)
            matAux[x][y] =  float(y1)/float(x1)
            y = y + 1;
        x = x + 1;
        y = 0;

    #print matAux
    #print matrixProbDocTerms
    matrixProbTermsDoc = matrixProbDocTerms.transpose()*matAux
    matrixProbTermsDoc = matrixProbTermsDoc/numpy.array([matrixProbTermsDoc.sum(axis=1)]).transpose()
    #print matrixProbTermsDoc
    #print matrixDocTermFinal

    rows = matrixProbTermsDoc.shape[0]
    cols = matrixProbTermsDoc.shape[1]




    matrixTermDocOrig = np.zeros((numTerm,numDocs))
    #print Prob
    for i in range(rows):
        for j in range(cols):
            #print Orig.item(i,j)
            if matrixProbTermsDoc.item(i,j) > 0:
                matrixTermDocOrig[i,j] = 1;

    rows = matrixTermDocOrig.shape[0]
    cols = matrixTermDocOrig.shape[1]

    print "cols"
    #print cols

    trivProv =  float(1/float(cols))



    DTMTrivial = numpy.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            DTMTrivial[i][j] = trivProv;

    print "===FIN Funcion LDAPerplexity==="
    return matrixTermDocOrig, matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, matrixPriorsCorpus[0], priorsDoc;

def calculatePerplexity(DMTO,DTMEmp,DTMProb):
    import numpy



    print DTMEmp.shape[0]
    print DTMEmp.shape[1]

    rows = DMTO.shape[0]
    cols = DMTO.shape[1]

    trivProv =  float(1/float(cols))
    DTMTrivial = numpy.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            DTMTrivial[i][j] = trivProv;



    termsPerDoc = []

    probVectorTrivial= []
    probVectorEmp= []
    probVectorProb= []
    probVectorOrig= []


    N_p = 0

    multTrivial = 1;
    multEmp = 1;
    multProb = 1;
    multOrig = 1


    for j in range(cols):
        for i in range(rows):
            if DMTO.item(i,j) == 1:
                multTrivial = multTrivial * DTMTrivial.item(i,j)
                multEmp =  multEmp * DTMEmp.item(i,j);
                multProb = multProb * DTMProb.item(i,j)
                multOrig = multOrig * DMTO.item(i,j)
                N_p = N_p + 1;

        termsPerDoc.append(N_p);


        if multTrivial == 0.0:
            multTrivial = 1e-300
        probVectorTrivial.append(math.exp(-1*math.log(multTrivial)/N_p))
        if multEmp == 0.0:
            multEmp = 1e-300
        probVectorEmp.append(math.exp(-1*math.log(multEmp)/N_p))
        if multProb == 0.0:
            multProb = 1e-300
        probVectorProb.append(math.exp(-1*math.log(multProb)/N_p))
        if multOrig == 0.0:
            multOrig = 1e-300
        probVectorOrig.append(math.exp(-1*math.log(multOrig)/N_p))


        #probVector.append(math.log(mult))

        N_p = 0
        multTrivial = 1;
        multEmp = 1;
        multProb = 1;
        multOrig = 1


    #print termsPerDoc
    #print "Perp_Triv","\t","Perp_Prob","\t","Perp_Orig"
    #print sum(probVectorTrivial),"\t", sum(probVectorProb),"\t", sum(probVectorOrig)
    #print "Perplexity Emp"
    return sum(probVectorProb), sum(probVectorEmp), probVectorProb, probVectorEmp

def calculateDistance(DTMO,DTMEmp,DTMProb):
    import numpy

    #print DTMO
    #print DTMProb
    rows = DTMO.shape[0]
    cols = DTMO.shape[1]

    DTMTrivial = numpy.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            DTMTrivial[i][j] = 0.5;


    sumOriginalTrivial = 0
    sumProbTrivial = 0
    sumEmpiricaTrivial = 0;
    sumOriginalEmp = 0
    sumProbEmp = 0
    sumTrivial = 0
    sumO = 0
    sumProb = 0
    sumEmp = 0

    sum2Trivial = 0;

    for i in range(rows):
        for j in range(cols):
            #Entre la Original 1-0 y la Trivial
            sumOriginalTrivial = math.pow(DTMO.item(i,j) - DTMTrivial.item(i,j),2) + sumOriginalTrivial;
            #Entre la Probabilidad y la Trivial
            sumProbTrivial = math.pow(DTMProb.item(i,j) - DTMTrivial.item(i,j),2) + sumProbTrivial;
            #Entre la Empirica y la Trivial
            sumEmpiricaTrivial = math.pow(DTMEmp.item(i,j) - DTMTrivial.item(i,j),2) + sumEmpiricaTrivial;

            #Entre la Original 1-0 y la Empirica
            sumOriginalEmp = math.pow(DTMO.item(i,j) - DTMEmp.item(i,j),2) + sumOriginalEmp;
            #Entre la Probabilidad y la Empirica
            sumProbEmp = math.pow(DTMProb.item(i,j) - DTMEmp.item(i,j),2) + sumProbEmp;
            #Suma Trivial
            sumTrivial = math.pow(DTMTrivial.item(i,j),2) + sumTrivial;
            #Suma Original
            sumO = math.pow(DTMO.item(i,j),2) + sumO;
            #Suma Probabilidad
            sumProb = math.pow(DTMProb.item(i,j),2) + sumProb;
            #Suma Empirica
            sumEmp = math.pow(DTMEmp.item(i,j),2) + sumEmp

    #print math.sqrt(sum);print math.sqrt(sum2); print math.sqrt(sum3)
    #print math.sqrt(sumEmp);print math.sqrt(sum2Emp); print math.sqrt(sum3)

    #print "Dist-Orig-Emp","\t","Dist-Orig-Triv","\t","Dist-Emp-Triv"
    #print math.sqrt(sumOriginalEmp)/(math.sqrt(sumO)*math.sqrt(sumEmp)),"\t",\
    #      math.sqrt(sumOriginalTrivial)/(math.sqrt(sumO)*math.sqrt(sumTrivial)),"\t",\
    #      math.sqrt(sumEmpiricaTrivial)/(math.sqrt(sumEmp)*math.sqrt(sumTrivial))
    #print math.sqrt(sumOriginalEmp),"\t",\
    #      math.sqrt(sumOriginalTrivial),"\t",\
    #      math.sqrt(sumEmpiricaTrivial)
    #print "=="
    #print "Dist-Prob-Emp","\t","Dist-Prob-Triv","\t","Dist-Emp-Triv"
    #print math.sqrt(sumProbEmp)/(math.sqrt(sumProb)*math.sqrt(sumEmp)),"\t",\
    #      math.sqrt(sumProbTrivial)/(math.sqrt(sumProb)*math.sqrt(sumTrivial)),"\t",\
    #      math.sqrt(sumEmpiricaTrivial)/(math.sqrt(sumEmp)*math.sqrt(sumTrivial))
    #print math.sqrt(sumProbEmp),"\t",\
    #      math.sqrt(sumProbTrivial),"\t",\
    #      math.sqrt(sumEmpiricaTrivial)


    return math.sqrt(sumProbEmp)/(math.sqrt(sumProb)*math.sqrt(sumEmp))
    #return math.sqrt(sumProbEmp)

def generateSampleFileM(file,vectorNewProbabilities, method, iter, expmnt):

    f = open("C:/Mallet/testCMD/"+file,"r")
    vectorDocs = []
    docs = 0;
    for line in f:
        vectorDocs.append(line)
        docs = docs + 1;
    f.close();

    setProbabilities = [];
    vectorProbs = [];
    sum = 0.0;
    vectorProbs.append(sum);
    for doc, prob in zip(range(docs),vectorNewProbabilities):
        setProbabilities.append(prob);
        sum = sum + prob;
        vectorProbs.append(sum)



    print "VECTORPROBS"
    #print vectorProbs
    #print setProbabilities



    import time

    f = open("C:/Mallet/testCMD/"+file,"r")
    f2 = open("C:/Mallet/testCMD/"+str(file.split('\\')[0])+"/"+str(method)+"/RESULTS"+str(expmnt)+"/SampleFile"+str(iter)+".txt","w");

    d = 0;
    while d < docs:
        #print "d docs", d, docs
        time.sleep(0.2)
        random.seed(time.time());
        chDoc =  random.random()
        cont = 0;
        for elem in vectorProbs:

            if float(elem) == 0.0:

                continue;
            #print "float(elem-prob) , float(chDoc) ,  float(elem)"
            #print vectorNewProbabilities[cont]
            #print float("{0:.4f}".format(elem-vectorNewProbabilities[cont])), float("{0:.4f}".format(chDoc)), float("{0:.4f}".format(elem))

            if (float("{0:.4f}".format(elem-vectorNewProbabilities[cont])) < float("{0:.4f}".format(chDoc))) and (float("{0:.4f}".format(chDoc)) <  float("{0:.4f}".format(elem))):
                if "\n" in str(vectorDocs[cont]):
                    f2.write(vectorDocs[cont])
                    d = d + 1;
                else:
                    f2.write(vectorDocs[cont]+"\n")
                    d = d + 1;
                break;

            cont = cont+ 1;

    f.close()
    f2.close();

    #import sys; sys.exit("Error message")

    f2 = open("C:/Mallet/testCMD/"+str(file.split('\\')[0])+"/"+str(method)+"/RESULTS"+str(expmnt)+"/SampleFile"+str(iter)+".txt","r");
    f3 = open("C:/Mallet/testCMD/"+str(file.split('\\')[0])+"/"+str(method)+"/RESULTS"+str(expmnt)+"/SampleFileB.txt","w");

    for elem in f2:
        #print elem
        f3.write(elem);
    f2.close();
    f3.close()
    os.remove("C:/Mallet/testCMD/"+str(file.split('\\')[0])+"/"+str(method)+"/RESULTS"+str(expmnt)+"/SampleFile"+str(iter)+".txt")

    f3 = open("C:/Mallet/testCMD/"+str(file.split('\\')[0])+"/"+str(method)+"/RESULTS"+str(expmnt)+"/SampleFileB.txt","r");
    f2 = open("C:/Mallet/testCMD/"+str(file.split('\\')[0])+"/"+str(method)+"/RESULTS"+str(expmnt)+"/SampleFile"+str(iter)+".txt","w");

    cont = 0;
    for elem in f3:
        #print elem
        if cont == 0:
            f2.write(str(elem).replace("D T:","D1 T:"));
        else:
            f2.write(str(elem).replace("D1 T:","D T:"));
        cont = cont + 1;

    f2.close(); f3.close();
    os.remove("C:/Mallet/testCMD/"+str(file.split('\\')[0])+"/"+str(method)+"/RESULTS"+str(expmnt)+"/SampleFileB.txt")

    #import sys; sys.exit("Error message")
    return str(str(file.split('\\')[0])+"/"+str(method)+"/RESULTS"+str(expmnt)+"/SampleFile"+str(iter)+".txt"), 0, setProbabilities

def generateInverseConditional(OrigCondMatrix, PriorA, PriorB):

    #Orig (P(A/B))
    #Inv P(B/A)= P(A/B)*(P(B)/P(A)

    matAux = numpy.zeros((len(PriorB),len(PriorA)))

    print len(PriorA)
    print len(PriorB)
    print PriorA
    print PriorB

    x = 0; y = 0;
    for pB in PriorB:
        for pA in PriorA:
            matAux[x][y] = float(float(pA)/float(pB))


            y = y + 1;
        x = x + 1;
        y = 0;



    #print OrigCondMatrix.transpose()*matAux

    #print OrigCondMatrix.transpose()

    #print matAux
    OrigCondMatrix = OrigCondMatrix + 0.00000000000000001
    OrigCondMatrix = OrigCondMatrix/numpy.array([OrigCondMatrix.sum(axis=1)]).transpose()

    InvCondMatrix = OrigCondMatrix.transpose()*matAux
    print InvCondMatrix



    InvCondMatrix = InvCondMatrix/numpy.array([InvCondMatrix.sum(axis=1)]).transpose()

    print len(numpy.array([InvCondMatrix.sum(axis=1)])[0])
    print numpy.array([InvCondMatrix.sum(axis=1)])

    for x in range(InvCondMatrix.shape[0]):
        for y in range(InvCondMatrix.shape[1]):
            if  math.isnan(float(InvCondMatrix[x][y])):
                print x,y
                print float(InvCondMatrix[x][y]), OrigCondMatrix.transpose()[x][y], matAux[x][y]

    #import sys; sys.exit("Error message")

    return InvCondMatrix

def KLD(DTMEmp, DTMReal):
    print "===Inicio KLD==="
    sumaTotal = 0.0
    for Q, P in zip(DTMEmp, DTMReal):
        suma= 0.0;
        for q, p in zip(Q,P):
            if p != 0.0:
                suma = suma + p*(math.log(p/q));

        sumaTotal = sumaTotal + suma
        #print suma
    print sumaTotal
    return sumaTotal

    print "===FIN KLD==="

def saveFileMatrix(matrix,matrixName,iter,file,method):

    rows = matrix.shape[0];
    cols = matrix.shape[1];

    f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/ModelMatrix/"+str(matrixName)+"_"+str(iter)+".txt","w");

    for fr in range(rows):
        for fc in range(cols):
            f.write(str(matrix[fr][fc])+"\t")
        f.write("\n")
    f.close()

def saveFileList(list,listName,iter,file,method):

    f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/ModelMatrix/"+str(listName)+"_"+str(iter)+".txt","w");

    for elem in list:
            f.write(str(elem)+"\t")
    f.close()

def trainPerplexity(trainWords, testWords, testDocs, iterations, file):
    print "===INICIO  NEW testCorpusPerplexity==="
    listaTema =[]
    numTopics = 0;
    for line in range(iterations):
        f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/ModelMatrix/MATRIXTOPWORD_"+str(line)+".txt","r");
        for line2 in f:
            numTopics = numTopics + 1;
        f.close()
    print numTopics, "numTopics"

    testWordsIndex = [];
    for elem in testWords:
        cont = 0
        for elem2 in trainWords:
            if elem == elem2:
                testWordsIndex.append(cont)
                break;
            else:
                cont = cont+ 1;

    print testWordsIndex

    topicPriors = float(1/float(numTopics))


    wordsNumber = len (testWordsIndex)

    sumaPerp = 0;
    indx = 1;
    for indx in testWordsIndex:
        linea = 0;
        for line in range(iterations):
            f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/ModelMatrix/MATRIXTOPWORD_"+str(line)+".txt","r");
            for line2 in f:

                print "indice palabra de _",indx," de ", wordsNumber,"palabras", "; linea ", linea,"de ",numTopics, "; archivo ", line

                listaTema = str(line2).split("\t"); del listaTema[-1]
                listaTema = [float(elem)+0.0000000000000001 for elem in listaTema]
                #listaTema = [float(i)/sum(listaTema) for i in listaTema]
                sumaPerp = math.log(float(listaTema[indx])*topicPriors) + sumaPerp
                linea = linea + 1;



            f.close()

    print sumaPerp

def trainPerplexityInv(trainWords, testWords, testDocs, iterations, file,method):
    print "===INICIO  NEW testCorpusPerplexity INVERTIDO==="
    listaTema =[]
    numTopics = 0;

    #print "iterations", range(iterations+1)
    for line in range(iterations+1):
        f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/ModelMatrix/MATRIXTOPWORD_"+str(line)+".txt","r");
        for line2 in f:
            numTopics = numTopics + 1;
        f.close()
    #print numTopics, "numTopics"

    testWordsIndex = [];
    for elem in testWords:
        cont = 0
        for elem2 in trainWords:
            if elem == elem2:
                testWordsIndex.append(cont)
                break;
            else:
                cont = cont+ 1;

    print testWordsIndex

    topicPriors = float(1/float(numTopics))


    wordsNumber = len (testWordsIndex)

    sumaPerp = 0;
    for line in range(iterations+1):
        f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/ModelMatrix/MATRIXTOPWORDinv_"+str(line)+".txt","r");
        linea = 0;
        for line2 in f:
            listaPalabra = [];
            if linea in testWordsIndex:
                #print "palabra ", linea

                listaPalabra = str(line2).split("\t"); del listaPalabra[-1]
                #print listaPalabra


                listaPalabra = [float(elem)+0.0000000000000001 for elem in listaPalabra]
                #print listaPalabra
                listaPalabra = [math.log(float(elem)*topicPriors) for elem in listaPalabra]
                #print "con sumar"
                #print listaPalabra
                sumaPerp =  sum(listaPalabra) + sumaPerp
                #print "indice palabra de _",indx," de ", wordsNumber,"palabras", "; linea ", linea,"de ",numTopics, "; archivo ", line

                #listaTema = str(line2).split("\t"); del listaTema[-1]
                #listaTema = [float(elem)+0.0000000000000001 for elem in listaTema]
                #listaTema = [float(i)/sum(listaTema) for i in listaTema]
                #sumaPerp = math.log(float(listaTema[indx])*topicPriors) + sumaPerp
            linea = linea + 1;



        f.close()

    print sumaPerp
    print "===FIN  NEW testCorpusPerplexity INVERTIDO==="
    return sumaPerp

def trainPerplexityInvMixedModel(trainWords, testWords, testDocs, iterations, file,method):
    print "===INICIO  NEW testCorpusPerplexity INVERTIDO==="
    listaTema =[]
    numTopics = 0;

    #print "iterations", range(iterations+1)
    for line in range(iterations+1):
        f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/ModelMatrix/MATRIXTOPWORD_MixedModel_"+str(line)+".txt","r");
        for line2 in f:
            numTopics = numTopics + 1;
        f.close()
    #print numTopics, "numTopics"

    testWordsIndex = [];
    for elem in testWords:
        cont = 0
        for elem2 in trainWords:
            if elem == elem2:
                testWordsIndex.append(cont)
                break;
            else:
                cont = cont+ 1;

    print testWordsIndex

    topicPriors = float(1/float(numTopics))


    wordsNumber = len (testWordsIndex)

    sumaPerp = 0;
    for line in range(iterations+1):
        f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/ModelMatrix/MATRIXTOPWORD_MixedModelinv_"+str(line)+".txt","r");
        linea = 0;
        for line2 in f:
            listaPalabra = [];
            if linea in testWordsIndex:
                #print "palabra ", linea

                listaPalabra = str(line2).split("\t"); del listaPalabra[-1]
                #print listaPalabra


                listaPalabra = [float(elem)+0.0000000000000001 for elem in listaPalabra]
                #print listaPalabra
                listaPalabra = [math.log(float(elem)*topicPriors) for elem in listaPalabra]
                #print "con sumar"
                #print listaPalabra
                sumaPerp =  sum(listaPalabra) + sumaPerp
                #print "indice palabra de _",indx," de ", wordsNumber,"palabras", "; linea ", linea,"de ",numTopics, "; archivo ", line

                #listaTema = str(line2).split("\t"); del listaTema[-1]
                #listaTema = [float(elem)+0.0000000000000001 for elem in listaTema]
                #listaTema = [float(i)/sum(listaTema) for i in listaTema]
                #sumaPerp = math.log(float(listaTema[indx])*topicPriors) + sumaPerp
            linea = linea + 1;



        f.close()

    print sumaPerp
    print "===FIN  NEW testCorpusPerplexity INVERTIDO==="
    return sumaPerp

def trainPerplexityInvGeneral(trainWords, testWords, testDocs, iterations, file,method, topicFile, InvTopicFile):
    print "===INICIO  NEW testCorpusPerplexity INVERTIDO==="
    listaTema =[]
    numTopics = 0;

    #print "iterations", range(iterations+1)
    #for line in range(iterations+1):
    f = open(topicFile,"r");
    for line2 in f:
        numTopics = numTopics + 1;
    f.close()
    print numTopics, "numTopics"

    testWordsIndex = [];
    for elem in testWords:
        cont = 0
        for elem2 in trainWords:
            if elem == elem2:
                testWordsIndex.append(cont)
                break;
            else:
                cont = cont+ 1;

    print testWordsIndex

    topicPriors = float(1/float(numTopics))


    wordsNumber = len (testWordsIndex)

    sumaPerp = 0;

    f = open(InvTopicFile,"r");
    linea = 0;
    for line2 in f:
        listaPalabra = [];
        if linea in testWordsIndex:
            #print "palabra ", linea

            listaPalabra = str(line2).split("\t"); del listaPalabra[-1]
            #print listaPalabra


            listaPalabra = [float(elem)+0.0000000000000001 for elem in listaPalabra]
            #print listaPalabra
            listaPalabra = [math.log(float(elem)*topicPriors) for elem in listaPalabra]
            #print "con sumar"
            #print listaPalabra
            sumaPerp =  sum(listaPalabra) + sumaPerp
            #print "indice palabra de _",indx," de ", wordsNumber,"palabras", "; linea ", linea,"de ",numTopics, "; archivo ", line

            #listaTema = str(line2).split("\t"); del listaTema[-1]
            #listaTema = [float(elem)+0.0000000000000001 for elem in listaTema]
            #listaTema = [float(i)/sum(listaTema) for i in listaTema]
            #sumaPerp = math.log(float(listaTema[indx])*topicPriors) + sumaPerp
        linea = linea + 1;



    f.close()

    print sumaPerp
    print "===FIN  NEW testCorpusPerplexity INVERTIDO==="
    return sumaPerp


dataset = "Dataset6_alt_atheism";
#method = "LDARep"
#method = "BAGGING_MIXEDMODEL"
method = "ADABOOST_MIXEDMODEL"

print "Debug No:1, Si:0 :"
debug = int(raw_input("Seleccione: "))

if dataset == "Dataset4":
    nombreData = "Dataset4Train"
    file = "Datastet_LDA_Toy_Cuatro\\TrainWordsPresentationExample.txt";
    testfile = "Datastet_LDA_Toy_Cuatro\\TrainWordsPresentationExampleTest.txt";
    numberOfTopics = [2];

#100 temas max
if dataset == "Dataset5":
    nombreData = "Dataset5Train50"
    file = "trainDOS\\50DocsTrain.txt";
    testfile = "trainDOS\\50DocsTest.txt";
    numberOfTopics = [2];

if dataset == "Dataset6_talk_religion_misc":
    nombreData = "Dataset6_talk_religion"
    file = "DS_talk_religion_misc\\talk_religion_misc_train.txt";
    testfile = "DS_talk_religion_misc\\talk_religion_misc_test.txt";
    numberOfTopics = [16];

if dataset == "Dataset6_talk_politics_misc":
    nombreData = "Dataset6_talk_politics"
    file = "DS_talk_politics_misc\\talk_politics_misc_train.txt";
    testfile = "DS_talk_politics_misc\\talk_politics_misc_test.txt";
    numberOfTopics = [16];

if dataset == "Dataset6_talk_politics_mideast":
    nombreData = "Dataset6_talk_politics_mideast"
    file = "DS_talk_politics_mideast\\talk_politics_mideast_train.txt";
    testfile = "DS_talk_politics_mideast\\talk_politics_mideast_test.txt";
    numberOfTopics = [24];

if dataset == "Dataset6_talk_politics_guns":
    nombreData = "Dataset6_talk_politics_guns"
    file = "DS_talk_politics_guns\\talk_politics_guns_train.txt";
    testfile = "DS_talk_politics_guns\\talk_politics_guns_test.txt";
    numberOfTopics = [20];

if dataset == "Dataset6_soc_religion_christian":
    nombreData = "Dataset6_soc_religion_christian"
    file = "DS_soc_religion_christian\\soc_religion_christian_train.txt";
    testfile = "DS_soc_religion_christian\\soc_religion_christian_test.txt";
    numberOfTopics = [28];

if dataset == "Dataset6_sci_space":
    nombreData = "Dataset6_sci_space"
    file = "DS_sci_space\\sci_space_train.txt";
    testfile = "DS_sci_space\\sci_space_test.txt";
    numberOfTopics = [24];

if dataset == "Dataset6_sci_med":
    nombreData = "Dataset6_sci_med"
    file = "DS_sci_med\\sci_med_train.txt";
    testfile = "DS_sci_med\\sci_med_test.txt";
    numberOfTopics = [18];

if dataset == "Dataset6_sci_electronics":
    nombreData = "Dataset6_sci_electronics"
    file = "DS_sci_electronics\\sci_electronics_train.txt";
    testfile = "DS_sci_electronics\\sci_electronics_test.txt";
    numberOfTopics = [12];

if dataset == "Dataset6_sci_crypt":
    nombreData = "Dataset6_sci_crypt"
    file = "DS_sci_crypt\\sci_crypt_train.txt";
    testfile = "DS_sci_crypt\\sci_crypt_test.txt";
    numberOfTopics = [18];

if dataset == "Dataset6_rec_sport_hockey":
    nombreData = "Dataset6_rec_sport_hockey"
    file = "DS_rec_sport_hockey\\rec_sport_hockey_train.txt";
    testfile = "DS_rec_sport_hockey\\rec_sport_hockey_test.txt";
    numberOfTopics = [18];

if dataset == "Dataset6_rec_sport_baseball":
    nombreData = "Dataset6_rec_sport_baseball"
    file = "DS_rec_sport_baseball\\rec_sport_baseball_train.txt";
    testfile = "DS_rec_sport_baseball\\rec_sport_baseball_test.txt";
    numberOfTopics = [18];

if dataset == "Dataset6_rec_motorcycles":
    nombreData = "Dataset6_rec_motorcycles"
    file = "DS_rec_motorcycles\\rec_motorcycles_train.txt";
    testfile = "DS_rec_motorcycles\\rec_motorcycles_test.txt";
    numberOfTopics = [12];

if dataset == "Dataset6_rec_autos": #AQUI VOY
    nombreData = "Dataset6_rec_autos"
    file = "DS_rec_autos\\rec_autos_train.txt";
    testfile = "DS_rec_autos\\rec_autos_test.txt";
    numberOfTopics = [18];

if dataset == "Dataset6_misc_forsale":
    nombreData = "Dataset6_misc_forsale"
    file = "DS_misc_forsale\\misc_forsale_train.txt";
    testfile = "DS_misc_forsale\\misc_forsale_test.txt";
    numberOfTopics = [12];

if dataset == "Dataset6_comp_windows_x":
    nombreData = "Dataset6_comp_windows_x"
    file = "DS_comp_windows_x\\comp_windows_x_train.txt";
    testfile = "DS_comp_windows_x\\comp_windows_x_test.txt";
    numberOfTopics = [12];

if dataset == "Dataset6_comp_sys_mac_hardware":
    nombreData = "Dataset6_comp_sys_mac_hardware"
    file = "DS_comp_sys_mac_hardware\\comp_sys_mac_hardware_train.txt";
    testfile = "DS_comp_sys_mac_hardware\\comp_sys_mac_hardware_test.txt";
    numberOfTopics = [12];

if dataset == "Dataset6_comp_sys_ibm_pc_hardware":
    nombreData = "Dataset6_comp_sys_ibm_pc_hardware"
    file = "DS_comp_sys_ibm_pc_hardware\\comp_sys_ibm_pc_hardware_train.txt";
    testfile = "DS_comp_sys_ibm_pc_hardware\\comp_sys_ibm_pc_hardware_test.txt";
    numberOfTopics = [12];

if dataset == "Dataset6_comp_os_ms_windows_misc":
    nombreData = "Dataset6_comp_os_ms_windows_misc"
    file = "DS_comp_os_ms_windows_misc\\comp_os_ms_windows_misc_train.txt";
    testfile = "DS_comp_os_ms_windows_misc\\comp_os_ms_windows_misc_test.txt";
    numberOfTopics = [14];

if dataset == "Dataset6_comp_graphics":
    nombreData = "Dataset6_comp_graphics"
    file = "DS_comp_graphics\\comp_graphics_train.txt";
    testfile = "DS_comp_graphics\\comp_graphics_test.txt";
    numberOfTopics = [16];

if dataset == "Dataset6_alt_atheism":
    nombreData = "Dataset6_alt_atheism"
    file = "DS_alt_atheism\\alt_atheism_train.txt";
    testfile = "DS_alt_atheism\\alt_atheism_test.txt";
    numberOfTopics = [20];



dataSetProbFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/DatasetProbabilities.txt","w");


f = open("C:/Mallet/testCMD/"+file,"r"); cont = 0;
for elem in f: cont = cont + 1;
f.close();
f = open("C:/Mallet/testCMD/"+file,"r");
for elem in f: dataSetProbFile.write(str(float(1/float(cont)))+"\n");
f.close();
dataSetProbFile.close();


#===========OBTENER PALABRAS DE ENTRENAMIENTO=================
numTopics = 2
alpha = 1;
beta = 1;

folder = file.split("\\")[0];
formatingFiles(numTopics,file,folder,alpha, beta)
[matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)


#import sys; sys.exit("Error message")


wordsFileY = open("C:/Mallet/testCMD/"+testfile.split("\\")[0]+"/words_count_file.txt","r");
trainWords = [];
for elem in wordsFileY: trainWords.append(elem.split(" ")[1])
wordsFileY.close(); #print trainWords

f3 = open("C:/Mallet/testCMD/"+testfile.split("\\")[0]+"/WordPerDoc.txt","r");
wordList = [];
docNumber = 0;
for line in f3:
    if "D1" in line: continue;
    if "target: T" in line: continue;
    if "name: D" in line: docNumber = docNumber + 1; continue;
    if len(line) == 1: continue;
    if "input: 0" in line:
        wordList.append(str(docNumber)+" " + line.split(" ")[2] + " " + line.split(" ")[3].replace("(","").replace(")",""));
    else:
        wordList.append(str(docNumber)+" " + line.split(" ")[1] + " " + line.split(" ")[2].replace("(","").replace(")",""));
f3.close(); #print wordList


#===========OBTENER DOCUMENTOS DE ENTRENAMIENTO=================
trainDocs = [];
trainDoc = [];
cont = 0
for elem in wordList:
    if int(elem.split(" ")[0]) == cont:
        trainDoc.append(elem.split(" ")[1]);
    else:
        trainDocs.append(trainDoc);
        trainDoc = [];
        trainDoc.append(elem.split(" ")[1])
        cont = cont + 1;
trainDocs.append(trainDoc)
#print trainDocs


#===========OBTENER PALABRAS DE TEST=================
folder = file.split("\\")[0];
formatingFilesTest(2,testfile,folder,alpha, beta)
wordsFileX = open("C:/Mallet/testCMD/"+testfile.split("\\")[0]+"/words_count_file-Test.txt","r");
testWords = [];
for elem in wordsFileX: testWords.append(elem.split(" ")[1])
wordsFileX.close()


f3 = open("C:/Mallet/testCMD/"+testfile.split("\\")[0]+"/WordPerDoc-Test.txt","r");
wordList = [];
docNumber = 0;
for line in f3:
    if "D1" in line: continue;
    if "target: T" in line: continue;
    if "name: D" in line: docNumber = docNumber + 1; continue;
    if len(line) == 1: continue;
    if "input: 0" in line:
        wordList.append(str(docNumber)+" " + line.split(" ")[2] + " " + line.split(" ")[3].replace("(","").replace(")",""));
    else:
        wordList.append(str(docNumber)+" " + line.split(" ")[1] + " " + line.split(" ")[2].replace("(","").replace(")",""));
f3.close(); #print wordList

#===========OBTENER DOCUMENTOS DE TEST=================
testDocs = [];
testDoc = [];
cont = 0
for elem in wordList:
    if int(elem.split(" ")[0]) == cont:
        testDoc.append(elem.split(" ")[1]);
    else:
        testDocs.append(testDoc);
        testDoc = [];
        testDoc.append(elem.split(" ")[1])
        cont = cont + 1;
testDocs.append(testDoc)


Words_for_Train = trainWords
Words_for_Test = testWords
Docs_for_Train_List = trainDocs
Docs_for_Test_List = testDocs

print "trainWords", len(trainWords)

import time

time.sleep(0.2)
random.seed(time.time());
numTopics  = random.choice(numberOfTopics);
num_top_words = 20;
numWords = len(trainWords)
numTestWords  =len(testWords)
TrainDocs = len(trainDocs)

alpha = float(50/float(numTopics));
beta = float(200/float(numWords));

if debug == 0: print trainDocs
if debug == 0: print trainWords
if debug == 0: print testDocs
if debug == 0: print testWords




iterations = 10

itNumber = iterations

trainValues = []
testValues = []
topicValues = []


numTopicsTotal= 0
topicPerIt = []
resultsFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/ResultsCluster.txt","w");
for it in range(iterations):
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
    f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+str(letters[it])+".txt","r");
    for l in f:
        numTopicsTotal = numTopicsTotal + 1;
    f.close()
    topicPerIt.append(str(numTopicsTotal))


resultsFile.write("total topics "+str(numTopicsTotal)+"\n")


for s_it in range(1,iterations+1):


    flag = 0
    nextClusterNumber = 0;


    #s_it = 9
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
    obj = slice(0,s_it)
    letters = letters[obj]

    print "#######################"
    print "########  VAMOS EN   ##", letters[obj]
    print "#######################"


    numTopicsInCluster = 0;
    numWordsInCluster = 0
    for it in range(s_it):
        f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+str(letters[it])+".txt","r");
        for line2 in f:
            numWordsInCluster = str(line2).split("\t"); del numWordsInCluster[-1]
            numWordsInCluster =  len(numWordsInCluster)
            numTopicsInCluster = numTopicsInCluster + 1;
        f.close()
    print "numWordsInCluster, ", numWordsInCluster, "numTopicsInCluster, ",numTopicsInCluster

    #continue;

    clusterTopicWordsMatrix = numpy.zeros((numTopicsInCluster,numWordsInCluster))

    x = 0;
    for it in range(s_it):
        f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+str(letters[it])+".txt","r");
        for line2 in f:
            numWordsInTopic = str(line2).split("\t"); del numWordsInTopic[-1]
            y = 0;
            for elem in numWordsInTopic:
                clusterTopicWordsMatrix[x][y] = float(elem)
                y = y + 1
            x = x + 1;
        f.close()


    numpy.random.seed(0)
    topicList = clusterTopicWordsMatrix.tolist()
    topicList = numpy.array(topicList)


    X = topicList
    #random.seed(random.seed(time.time()))
    #random.shuffle(X)
    #if debug == 0: print X

    if flag == 0:
        K = range(3,200,1)
    if flag == 1:
        K = range(3,nextClusterNumber,1);
    if debug == 0: print K
    if dataset == "Dataset4": K = [2,3]

    KM = list()
    centroids  =list()
    for k in K:
        #k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
        if len(X) > k:
            k_means = KMeans(n_clusters=k, random_state=1)
            #k_means = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=45, n_init=10, max_no_improvement=10, verbose=0)
            k_means.fit(X)

            #k_means_labels = k_means.labels_
            k_means_cluster_centers = k_means.cluster_centers_
            centroids.append(k_means_cluster_centers)

    #KM = [kmeans(X,k) for k in K]
    #centroids = [cent for (cent,var) in KM]


    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]  #DISTANCIA DE CADA ELEMENTO A CADA CENTROIDE
    cIdx = [np.argmin(D,axis=1) for D in D_k]   # CLUSTER AL QUE PERTENECE CADA ELEMENTO
    dist = [np.min(D,axis=1) for D in D_k]   # DISTANCIA DE CADA ELEMENTO A SU CENTROIDE
    avgWithinSS = [sum(d)/X.shape[0] for d in dist]   # PROMEDIO DE DISTANCIAS

    if debug == 0: print "D_k ",D_k
    if debug == 0: print "cIdx ", cIdx
    if debug == 0: print "dist ", dist
    print "avgWithinSS ", avgWithinSS



    x = 0
    sumList = []
    for elem in avgWithinSS:
        if x < len(avgWithinSS)-1:
            sumList.append(((avgWithinSS[x] - avgWithinSS[x+1])/avgWithinSS[x])*100)
            x = x + 1

    if debug == 0: print sumList
    if debug == 0: print np.min(sumList)



    min = float('inf')
    mins = []
    x = 0

    distThes = 20.0
    numPoints = 1;
    if dataset == "Dataset4":  distThes = 50.0; numPoints = 1;
    for n in range(numPoints):
        for elem in sumList:
            if float(elem) < distThes and float(elem) > 0.0:
                if float(elem) < min:
                    min = float(elem)
        if min != float('inf'):
            mins.append(sumList.index(min))
        else:
            mins.append(0)
            min=3
        sumList[sumList.index(min)] = 0.0;
        min = float('inf')


    if debug == 0: print sumList
    if debug == 0: print "min ", min
    if debug == 0: print "mins, ", mins



    nCentr = 0;
    matrixList = []
    if dataset =="Dataset4": mins = [0]

    for c in centroids:
        if nCentr in mins:
            numClusterTop = len(c)
            numClusterWords = len(c[0])
            matrixTotalTopicsWords = numpy.zeros((numClusterTop,numClusterWords))
            x = 0
            for el in c:
                y = 0
                for num in el:
                    matrixTotalTopicsWords[x][y] = float(num)
                    y = y + 1
                x = x + 1
            if debug == 0: print "x, ", x
            matrixList.append(matrixTotalTopicsWords)
        nCentr = nCentr + 1


    if debug == 0: print "nCentr ", nCentr
    if debug == 0: print matrixList




    fig = plt.figure()
    ax = fig.add_subplot(111)
    print "K ", len(K)
    K = K[:len(avgWithinSS)]
    print "K ", len(K), K

    print "avgWithinSS", len(avgWithinSS)
    ax.plot(K, avgWithinSS, 'b*-')


    #for el in mins:
    #    ax.plot(K[el], avgWithinSS[el], marker='o', markersize=12,
    #        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    #plt.grid(True)
    #plt.xlabel('Number of clusters')
    #plt.ylabel('Average within-cluster sum of squares')
    #plt.title('Elbow for KMeans clustering')
    #plt.show()




    for elem in matrixList:
        rows = elem.shape[0]; cols = elem.shape[1]
        clusterTopics = int(rows)
        matrixTopWord = elem

        topicFile = "C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/TMCluster_Results_TRAIN.txt"
        fileToWrite =  open(topicFile,"w");
        rows = matrixTopWord.shape[0]; cols = matrixTopWord.shape[1]
        for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
            for col in range(cols):
                fileToWrite.write(str(matrixTopWord.item(row,col))+"\t")
            fileToWrite.write("\n");
        fileToWrite.close();

        InvTopicFile = "C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/TMCluster_Results_TRAIN_inv.txt"
        fileToWrite =  open(InvTopicFile,"w");
        matrixTotalTopicsWordsInv = matrixTopWord.transpose()
        rows = matrixTotalTopicsWordsInv.shape[0]; cols = matrixTotalTopicsWordsInv.shape[1]
        for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
            for col in range(cols):
                fileToWrite.write(str(matrixTotalTopicsWordsInv.item(row,col))+"\t")
            fileToWrite.write("\n");
        fileToWrite.close();

        trainPPL = trainPerplexityInvGeneral(trainWords, trainWords, testDocs, itNumber, file,method, topicFile, InvTopicFile)
        testPPL = trainPerplexityInvGeneral(trainWords, testWords, testDocs, itNumber, file,method, topicFile, InvTopicFile)

        print "train ", float(float(trainPPL)/float(clusterTopics))
        print "test ", float(float(testPPL)/float(clusterTopics))
        print "numTopY ", clusterTopics

        trainValues.append(str(float(float(trainPPL)/float(clusterTopics))))
        testValues.append(str(float(float(testPPL)/float(clusterTopics))))
        topicValues.append(str(clusterTopics))



    centroidsTopicWordsMatrix = matrixList[0]
    print centroidsTopicWordsMatrix.shape[0], centroidsTopicWordsMatrix.shape[1]

    print numpy.array([centroidsTopicWordsMatrix.sum(axis=1)])

resultsFile.write('\t'.join(trainValues).replace(".",",")+"\n")
resultsFile.write('\t'.join(testValues).replace(".",",")+"\n")
resultsFile.write('\t'.join(topicValues).replace(".",",")+"\n")
resultsFile.write('\t'.join(topicPerIt).replace(".",",")+"\n")
resultsFile.close()


print trainValues
print testValues
print topicValues


import sys; sys.exit("Error message")

        #
        # #### ==== OBTENIENDO DATOS PARA MATRIZ DOCUMENTO-TOPICOS ==== ####
        #
        # numDocsInCluster = 0
        # for it in range(s_it):
        #     f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+str(letters[it])+".txt","r");
        #     for line2 in f:
        #         numTopicsInCluster = str(line2).split("\t"); del numTopicsInCluster[-1]
        #         numTopicsInCluster =  len(numTopicsInCluster)
        #         numDocsInCluster = numDocsInCluster + 1;
        #     f.close()
        # print "numTopicsInCluster ", numTopicsInCluster
        # print "numDocsInCluster: ", numDocsInCluster
        #
        #
        # clusterDocTopicMatrix = numpy.zeros((numDocsInCluster,numTopicsInCluster))
        # x = 0;
        # for it in range(s_it):
        #     f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+str(letters[it])+".txt","r");
        #     print "C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+str(letters[it])+".txt"
        #     print "C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+str(letters[it])+".txt"
        #     for line2 in f:
        #         numWordsInTopic = str(line2).split("\t");
        #         del numWordsInTopic[-1]
        #         print len(numWordsInTopic)
        #         print numWordsInTopic
        #         y = 0;
        #         if len(numWordsInTopic) == numTopicsInCluster:
        #             for elem in numWordsInTopic:
        #                 clusterDocTopicMatrix[x][y] = float(elem)
        #                 y = y + 1
        #         x = x + 1;
        #     f.close()
        #
        #
        # docPriors = [float(1/float(numDocsInCluster))]*numDocsInCluster
        # topicPriors = [float(1/float(numTopicsInCluster))]*numTopicsInCluster
        #
        # clusterTopicDocMatrix = generateInverseConditional(clusterDocTopicMatrix,docPriors, topicPriors)
        #
        # minClusters = int(centroidsTopicWordsMatrix.shape[0]);
        #
        # topicList = clusterTopicDocMatrix.tolist()
        # topicList = numpy.array(topicList)
        #
        #
        # X = topicList
        # random.seed(random.seed(time.time()))
        # random.shuffle(X)
        # if debug == 0: print X
        #
        #
        # K = [minClusters]
        #
        # print "K, ", K
        #
        #
        # if debug == 0: print K
        # if dataset == "Dataset4": K = [2,3]
        #
        # print "len(X)" , len(X)
        # k_means = KMeans(n_clusters=K[0], random_state=1)
        # k_means.fit(X)
        # k_means_cluster_centers = k_means.cluster_centers_
        # centroids.append(k_means_cluster_centers)
        #
        #
        #
        #
        # #KM = [kmeans(X,k) for k in K]
        # #centroids = [cent for (cent,var) in KM]   # cluster centroids
        #
        # matrixList = []
        #
        #
        #
        # for c in centroids:
        #     numClusterTop = len(c)
        #     if debug == 0: print "c", len(c)
        #     numClusterDocs = len(c[0])
        #     clusterTotalTopicsDocsMatrix = numpy.zeros((numClusterTop,numClusterDocs))
        #     x = 0
        #     for el in c:
        #         y = 0
        #         for num in el:
        #             clusterTotalTopicsDocsMatrix[x][y] = float(num)
        #             y = y + 1
        #         x = x + 1
        #     if debug == 0: print "x, ", x
        #     matrixList.append(clusterTotalTopicsDocsMatrix)
        #
        #
        #
        # centroidsTopicsDocsMatrix = clusterTotalTopicsDocsMatrix
        #
        # print "#######################"
        # print "#######################"
        # print centroidsTopicWordsMatrix.shape[0], centroidsTopicWordsMatrix.shape[1]
        #
        # print centroidsTopicsDocsMatrix.shape[0], centroidsTopicsDocsMatrix.shape[1]
        # print "#######################"
        # print "#######################"
        #
        #
        #
        # if centroidsTopicWordsMatrix.shape[0] == centroidsTopicsDocsMatrix.shape[0]:
        #     numTopicsInCluster = centroidsTopicsDocsMatrix.shape[0]
        #     numDocsInCluster = centroidsTopicsDocsMatrix.shape[1]
        #
        #     docPriors = [float(1/float(numDocsInCluster))]*numDocsInCluster
        #     topicPriors = [float(1/float(numTopicsInCluster))]*numTopicsInCluster
        #
        #     clusterTotalDocsTopicsMatrix = generateInverseConditional(centroidsTopicsDocsMatrix,topicPriors, docPriors)
        #
        #     print centroidsTopicsDocsMatrix.shape[0], centroidsTopicsDocsMatrix.shape[1]
        #     print clusterTotalDocsTopicsMatrix.shape[0], clusterTotalDocsTopicsMatrix.shape[1]
        #
        #     clusterDocTermMatrix =  np.dot(clusterTotalDocsTopicsMatrix,centroidsTopicWordsMatrix)
        #
        #     success = True;
        #
        # else:
        #
        #     nextClusterNumber = int(centroidsTopicsDocsMatrix.shape[0]);
        #
        #
        #     flag = 1;
        #     success = False;
        #
        # if success == True:
        #
        #     print "xxxxxxxxxxxxxxxxxxxxxx"
        #     print clusterDocTermMatrix.shape[0]
        #     print clusterDocTermMatrix.shape[1]
        #     print len(priorsCorpus)
        #
        #     print len(priorsDoc)
        #     print "xxxxxxxxxxxxxxxxxxxxxx"
        #     #print priorsCorpus
        #     #print priorsDoc
        #
        #
        #     clusterTermsDocsMatrix = generateInverseConditional(clusterDocTermMatrix,docPriors, priorsCorpus)
        #     #print clusterTermsDocsMatrix
        #     #print clusterTermsDocsMatrix.shape[0], clusterTermsDocsMatrix.shape[1]
        #     #print matrixDocTerms.shape[0], matrixDocTerms.shape[1]
        #     #print matrixProbTermsDoc.shape[0], matrixProbTermsDoc.shape[1]
        #
        #
        #
        #
        #
        #     [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(matrixDocTerms,clusterTermsDocsMatrix,matrixProbTermsDoc)
        #
        #     print "Perplexity_Prob_Final, Perplexity_Emp_Final"
        #     print Perplexity_Prob_Final, Perplexity_Emp_Final
        #
        #     #perpFile.write("FINAL: "+str(Perplexity_Prob_Final)+"\t"+str(Perplexity_Emp_Final)+"\t"+str(Perplexity_Prob_Final-Perplexity_Emp_Final)+"\n")
        #
        #
        #     Distance_Emp_Final = calculateDistance(matrixDocTerms,clusterTermsDocsMatrix,matrixProbTermsDoc)
        #
        #     print "Distance_Emp_Final"
        #     print Distance_Emp_Final
        #     #import sys; sys.exit("Error message")
        #
        #



for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()
#print "k_means_labels"
#print k_means_labels
#print "k_means_cluster_centers"
#print k_means_cluster_centers
import sys; sys.exit("Error message")

#mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
#t0 = time.time()
#mbk.fit(topicList)
#t_mini_batch = time.time() - t0
#mbk_means_labels = mbk.labels_
#mbk_means_cluster_centers = mbk.cluster_centers_
#mbk_means_labels_unique = np.unique(mbk_means_labels)

#silhouette_avg = silhouette_score(topicList, mbk_means_labels)
#sample_silhouette_values = silhouette_samples(topicList, mbk_means_labels)

#print "t_mini_batch"
#print t_mini_batch
#print "silhouette_avg"
#print silhouette_avg
#print "sample_silhouette_values"
#print sample_silhouette_values
#print "mbk_means_labels"
#print mbk_means_labels
#print "mbk.cluster_centers_"
#print mbk.cluster_centers_


#toplist = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
####toplist = [2,3]

####fileToWrite =  open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/BAGGING_MIXEDMODEL/SilhouettePerKClusterResults.txt","w");

####for cl in toplist:
####    n_clusters = cl
####    batch_size = 45
####    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
####    t0 = time.time()
####    k_means.fit(topicList)
####    t_batch = time.time() - t0
####    k_means_labels = k_means.labels_
####    k_means_cluster_centers = k_means.cluster_centers_
####    k_means_labels_unique = np.unique(k_means_labels)

####    silhouette_avg = silhouette_score(topicList, k_means_labels)
####    sample_silhouette_values = silhouette_samples(topicList, k_means_labels)

    #print "t_batch"
    #print t_batch
    #print "silhouette_avg"
    #print silhouette_avg
    #print "sample_silhouette_values"
    #print sample_silhouette_values
    #print "k_means_labels"
    #print k_means_labels
    #print "silhouette variance"
    #print numpy.var(sample_silhouette_values)
    #print "silhouette std"
    #print numpy.std(sample_silhouette_values)
    #print "cl ","\t", cl,"\t", " sil ","\t", silhouette_avg,"\t", " sil_std ","\t",numpy.std(sample_silhouette_values)

####    fileToWrite.write(str(str(cl)+"\t"+str(silhouette_avg)+"\t"+str(numpy.std(sample_silhouette_values))+"\n").replace(".",","))


####fileToWrite.close()


