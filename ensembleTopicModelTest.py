__author__ = 'Ormeno'

#from nltk.book import *

import os
import math
import random
import numpy
import math



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


    num_top_words = 10;

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

    num_top_words = 30;

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
        print lines
        f2.write(lines);
    for lines in dir3:
        f3.write(lines);
    for lines in dir4:
        f4.write(lines);

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    print "llego aca"

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



    x = 0; y = 0;
    for pB in PriorB:
        for pA in PriorA:
            matAux[x][y] = pA/pB


            y = y + 1;
        x = x + 1;
        y = 0;


    #print OrigCondMatrix.transpose()*matAux

    #print OrigCondMatrix.transpose()

    #print matAux

    InvCondMatrix = OrigCondMatrix.transpose()*matAux
    InvCondMatrix = InvCondMatrix/numpy.array([InvCondMatrix.sum(axis=1)]).transpose()
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


def BAGGING(numTopics, numWords, alpha, beta, itNumber=1):
    TOTAL_DTMTrivial = []; TOTAL_DTMO = []; TOTAL_DTMProb = []
    TOTAL_DTMEmp = []; TOTAL_TOP_WORD_Emp = []; TOTAL_TOP_PRIORS = [];
    TOTAL_DOC_TOP_Emp = [];

    import numpy
    f = open("C:/Mallet/testCMD/"+file,"r")
    docs = 0;
    for line in f: docs = docs + 1;
    f.close();
    prob = float(1/float(docs))
    setProbabilities = [];
    for doc in range(docs): setProbabilities.append(prob);

    iterations = itNumber;    topicsTotal = 0;
    for i in range(iterations):
        [sampleFile,number, setProbabilities] = generateSampleFileM(file, setProbabilities, "BAGGING")
        print sampleFile, number, setProbabilities

        folder = file.split("\\")[0];

        print "n_topics","alpha","beta"
        print numTopics, alpha, beta

        topicsTotal = topicsTotal + numTopics;
        formatingFiles(numTopics,file,folder,alpha, beta)
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)


        matrixWordTop = matrixTopWord.transpose()

        topicPriors = [float(1/float(numTopics))]*numTopics
        topicPriors = list(topicPriors)

        print "topicPriors====", topicPriors
        matrixTopWord = generateInverseConditional(matrixWordTop,priorsCorpus, topicPriors)

        saveFileMatrix(matrixTopWord.transpose(),"MATRIXTOPWORDinv",i,file,"BAGGING");
        saveFileMatrix(matrixTopWord,"MATRIXTOPWORD",i,file,"BAGGING");
        saveFileMatrix(matrixDocTop,"MATRIXDOCTOP",i,file,"BAGGING")
        saveFileMatrix(matrixProbTermsDoc,"MATRIXPROBTERMSDOCS",i,file,"BAGGING")
        saveFileMatrix(matrixDocTermFinal,"MATRIXDOCTERMSFINAL",i,file,"BAGGING")

        saveFileList(priorsCorpus,"PRIORSCORPUS",i,file,"BAGGING")
        saveFileList(topicPriors,"TOPICPRIORS",i,file,"BAGGING")


        TOTAL_DTMTrivial.append(DTMTrivial)
        TOTAL_DTMO.append(matrixDocTerms)
        TOTAL_DTMProb.append(matrixProbTermsDoc)
        TOTAL_DTMEmp.append(matrixDocTermFinal)
        rows = matrixTopWord.transpose().shape[0]; cols = matrixTopWord.transpose().shape[1]
        TOTAL_TOP_PRIORS.append(numpy.array([matrixTopWord.sum(axis=1)])/rows);
        TOTAL_TOP_WORD_Emp.append(matrixTopWord.transpose())

        print "Calculo Perplexity"
        [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)

        perpPerDocFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/BAGGING/PerplexityPerDoc"+str(i)+".txt","w");
        for l1, l2 in zip(probVectorProb, probVectorEmp):
            perpPerDocFile.write(str(l1)+"-"+str(l2)+"="+str(l1-l2)+"\n");
        perpPerDocFile.close();


        #perpFile.write(str(Perplexity_Prob)+"\t"+str(Perplexity_Emp)+"\t"+str(Perplexity_Prob-Perplexity_Emp)+"\n")

        print "Calculo Distancia"
        Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
        #print Distance_Emp
        #distFile.write(str(Distance_Emp)+"\n")

        print "it" + str(i)

    wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
    trainWords = [];
    for elem in wordsFileE: trainWords.append(elem.split(" ")[1])
    wordsFileE.close()


    trainPerp = trainPerplexityInv(trainWords, trainWords, testDocs, iterations, file,"BAGGING")

    testPerp = trainPerplexityInv(trainWords, testWords, testDocs, iterations, file,"BAGGING")


    #---import sys; sys.exit("Error message")

    MEAN_TOTAL_DTMProb = sum(TOTAL_DTMProb)/iterations
    MEAN_TOTAL_DTMEmp = sum(TOTAL_DTMEmp)/iterations

    print iterations
    #import sys; sys.exit("Error message")
    [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(sum(TOTAL_DTMO)/iterations,sum(TOTAL_DTMEmp)/iterations,sum(TOTAL_DTMProb)/iterations)

    print "Perplexity_Prob_Final, Perplexity_Emp_Final"
    print Perplexity_Prob_Final, Perplexity_Emp_Final

    #perpFile.write("FINAL: "+str(Perplexity_Prob_Final)+"\t"+str(Perplexity_Emp_Final)+"\t"+str(Perplexity_Prob_Final-Perplexity_Emp_Final)+"\n")

    Distance_Emp_Final = calculateDistance(sum(TOTAL_DTMO)/iterations,sum(TOTAL_DTMEmp)/iterations,sum(TOTAL_DTMProb)/iterations)

    print "Distance_Emp_Final"
    print Distance_Emp_Final



    matrixDocTermEmp_plot = sum(TOTAL_DTMEmp)/iterations;
    matrixDocTermProb_plot = sum(TOTAL_DTMProb)/iterations
    matrixTopWords_plot = sum(TOTAL_TOP_WORD_Emp)/iterations;



    kld = KLD(MEAN_TOTAL_DTMEmp, MEAN_TOTAL_DTMProb)

    print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld
    return trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld


def ADABOOST(numTopics, numWords, alpha, beta, itNumber=1,expmnt=1):

    TOTAL_DTMTrivial = []; TOTAL_DTMO = []; TOTAL_DTMProb = []
    TOTAL_DTMEmp = []; TOTAL_TOP_WORD_Emp = []; TOTAL_TOP_PRIORS = [];
    TOTAL_DOC_TOP_Emp = [];

    import time
    import glob
    import numpy

    f = open("C:/Mallet/testCMD/"+file,"r")
    docs = 0;
    for line in f: docs = docs + 1;
    f.close();
    prob = float(1/float(docs))
    setProbabilities = [];
    for doc in range(docs): setProbabilities.append(prob);
    [sampleFile,number, setProbabilities] = generateSampleFileM(file, setProbabilities, "ADABOOST", 0, expmnt)



    folder = file.split("\\")[0];

    machinesPerplexity = []; confidenceParametersAlphas = []; confidenceAlpha = 0;

    iterations = itNumber;

    for i in range(iterations):

        print "==ITERACION: ", i, " =="

        if i >= 1:
            [sampleFile,number, setProbabilities] = generateSampleFileM(file, vectorNewProbabilities, "ADABOOST",i,expmnt)
            print sampleFile, number, setProbabilities, vectorNewProbabilities

        successful = False;

        while not successful:
            print "Numero Topicos, Rounds"
            print numTopics, alpha, beta
            #numTopics = 3;


            print "Entrenando con Muestra";
            formatingFiles(numTopics,sampleFile,folder,alpha, beta)
            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,sampleFile,folder)
            [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)


            Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)

            #print "Distance_Emp"
            #print Distance_Emp

            confidenceAlpha  = float(0.5*math.log((1-Distance_Emp)/Distance_Emp));
            successful = True;
            confidenceParametersAlphas.append(confidenceAlpha);

            print "confidenceAlpha"
            print confidenceAlpha


        print "Probando con conjunto de Training"
        print "numTopics, alpha, beta"
        print numTopics, alpha, beta

        formatingFiles(numTopics,file,folder,alpha, beta)
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)


        while 0 in matrixTopWord.sum(axis=1):
            print "falla, probando de nuevo"
            formatingFiles(numTopics,file,folder,alpha, beta)
            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

        matrixWordTop = matrixTopWord.transpose()

        topicPriors = [float(1/float(numTopics))]*numTopics
        topicPriors = list(topicPriors)

        #print "topicPriors====", topicPriors
        matrixTopWord = generateInverseConditional(matrixWordTop,priorsCorpus, topicPriors)


        saveFileMatrix(matrixTopWord.transpose(),"MATRIXTOPWORDinv",i,file,"ADABOOST");
        saveFileMatrix(matrixTopWord,"MATRIXTOPWORD",i,file,"ADABOOST");
        saveFileMatrix(matrixDocTop,"MATRIXDOCTOP",i,file,"ADABOOST")
        saveFileMatrix(matrixProbTermsDoc,"MATRIXPROBTERMSDOCS",i,file,"ADABOOST")
        saveFileMatrix(matrixDocTermFinal,"MATRIXDOCTERMSFINAL",i,file,"ADABOOST")

        saveFileList(priorsCorpus,"PRIORSCORPUS",i,file,"ADABOOST")
        saveFileList(topicPriors,"TOPICPRIORS",i,file,"ADABOOST")



        [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
        #print "Perp Maquina: ", Perplexity_Prob
        machinesPerplexity.append(Perplexity_Emp);

        #print "Calculo Distancia"
        Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)

        vectorDifEmpProb  =[];
        vectorNewProbabilities = [];

        for l1, l2 in zip(probVectorEmp,probVectorProb):
            vectorDifEmpProb.append(float(l1-l2)/float(l1));

        for l1, l2 in zip(vectorDifEmpProb,setProbabilities):
            vectorNewProbabilities.append(float(l2)*math.exp(float(l1)*confidenceAlpha))


        sumProb = 0;

        for elem in vectorNewProbabilities:
            sumProb = sumProb + float(elem)

        vectorNewProbabilities[:] = [x/sumProb for x in vectorNewProbabilities]

        #print vectorNewProbabilities




        TOTAL_DTMO.append(matrixDocTerms)
        TOTAL_DTMProb.append(matrixProbTermsDoc)
        TOTAL_DTMEmp.append(matrixDocTermFinal)
        TOTAL_TOP_WORD_Emp.append(matrixTopWord.transpose());
        rows = matrixTopWord.transpose().shape[0]; cols = matrixTopWord.transpose().shape[1]
        TOTAL_TOP_PRIORS.append(numpy.array([matrixTopWord.sum(axis=1)])/rows);
        TOTAL_DOC_TOP_Emp.append(matrixDocTop);




    MEAN_TOTAL_DTMO = sum(TOTAL_DTMO)/iterations
    MEAN_TOTAL_DTMProb = sum(TOTAL_DTMProb)/iterations
    MEAN_TOTAL_DTMEmp = sum(TOTAL_DTMEmp)/iterations



    wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
    trainWords = [];
    for elem in wordsFileE: trainWords.append(elem.split(" ")[1])
    wordsFileE.close()


    trainPerp = trainPerplexityInv(trainWords, trainWords, testDocs, iterations, file,"ADABOOST")

    testPerp = trainPerplexityInv(trainWords, testWords, testDocs, iterations, file,"ADABOOST")




    #print iterations
    #import sys; sys.exit("Error message")

    [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(sum(TOTAL_DTMO)/iterations,sum(TOTAL_DTMEmp)/iterations,sum(TOTAL_DTMProb)/iterations)

    #print "Perplexity_Prob_Final, Perplexity_Emp_Final"
    #print Perplexity_Prob_Final, Perplexity_Emp_Final

    #perpFile.write("FINAL: "+str(Perplexity_Prob_Final)+"\t"+str(Perplexity_Emp_Final)+"\t"+str(Perplexity_Prob_Final-Perplexity_Emp_Final)+"\n")

    Distance_Emp_Final = calculateDistance(sum(TOTAL_DTMO)/iterations,sum(TOTAL_DTMEmp)/iterations,sum(TOTAL_DTMProb)/iterations)

    #print "Distance_Emp_Final"
    #print Distance_Emp_Final



    suma = 0;
    for l1, l2 in zip(confidenceParametersAlphas,TOTAL_DTMEmp):
        suma = suma + l1*l2
    #print suma

    suma =suma/numpy.array([suma.sum(axis=1)]).transpose()
    #print suma


    kld = KLD(MEAN_TOTAL_DTMEmp, MEAN_TOTAL_DTMProb)

    print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld
    return trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld

def LDA(numTopics, numWords, alpha, beta,itNumber=1,expmnt=1):


    import numpy
    folder = file.split("\\")[0];


    formatingFiles(numTopics,file,folder,alpha, beta)
    [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

    pruebas = 0;
    while 0 in matrixTopWord.sum(axis=1):
        pruebas = pruebas + 1;
        if pruebas > 5:
                return 0, 0, 0, 0, 0
        print "falla, probando de nuevo"
        print "Probando con ", numTopics, " topicos y alpha beta", alpha , beta
        formatingFiles(numTopics,file,folder,alpha, beta)
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)




    matrixWordTop = matrixTopWord.transpose()

    topicPriors = [float(1/float(numTopics))]*numTopics
    topicPriors = list(topicPriors)

    print "topicPriors====", topicPriors
    matrixTopWord = generateInverseConditional(matrixWordTop,priorsCorpus, topicPriors)
    #import sys; sys.exit("Error message")

    saveFileMatrix(matrixTopWord.transpose(),"MATRIXTOPWORDinv",0,file,"LDA");
    saveFileMatrix(matrixTopWord,"MATRIXTOPWORD",0,file,"LDA");
    saveFileMatrix(matrixDocTop,"MATRIXDOCTOP",0,file,"LDA")
    saveFileMatrix(matrixProbTermsDoc,"MATRIXPROBTERMSDOCS",0,file,"LDA")
    saveFileMatrix(matrixDocTermFinal,"MATRIXDOCTERMSFINAL",0,file,"LDA")

    saveFileList(priorsCorpus,"PRIORSCORPUS",0,file,"LDA")
    saveFileList(topicPriors,"TOPICPRIORS",0,file,"LDA")



    wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
    trainWords = [];
    for elem in wordsFileE: trainWords.append(elem.split(" ")[1])
    wordsFileE.close()



    trainPerp = trainPerplexityInv(trainWords, trainWords, testDocs,0, file,"LDA")

    testPerp = trainPerplexityInv(trainWords, testWords, testDocs, 0, file,"LDA")


    #trainPerplexity(trainWords, trainWords, trainDocs, iterations, file)
    #trainPerplexity(trainWords, testWords, testDocs, iterations, file)

    #import sys; sys.exit("Error message")


    [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
    print "Perplexity_Prob_Final, Perplexity_Emp_Final"
    print Perplexity_Prob_Final, Perplexity_Emp_Final



    Distance_Emp_Final = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
    print "Distance_Emp_Final"
    print Distance_Emp_Final




    #print matrixTopWord
    #testCorpusPerplexity(trainWords,trainWords, trainDocs,matrixTopWord)

    #testCorpusPerplexity(trainWords, testWords, testDocs, matrixTopWord);


    kld = KLD(matrixDocTermFinal,matrixProbTermsDoc)

    return trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld

def ADABOOST_V2(numTopics, numWords, alpha, beta, itNumber=2,expmnt=1):

    TOTAL_DTMTrivial = []; TOTAL_DTMO = []; TOTAL_DTMProb = []
    TOTAL_DTMEmp = []; TOTAL_TOP_WORD_Emp = []; TOTAL_TOP_PRIORS = [];
    TOTAL_DOC_TOP_Emp = [];

    import time
    import glob
    import numpy

    f = open("C:/Mallet/testCMD/"+file,"r")
    docs = 0;
    for line in f: docs = docs + 1;
    f.close();
    prob = float(1/float(docs))
    setProbabilities = [];
    for doc in range(docs): setProbabilities.append(prob);
    [sampleFile,number, setProbabilities] = generateSampleFileM(file, setProbabilities, "ADABOOST", 0, expmnt)



    folder = file.split("\\")[0];

    machinesPerplexity = []; confidenceParametersAlphas = []; confidenceAlpha = 0;

    iterations = itNumber;

    for i in range(iterations):

        print "==ITERACION: ", i, " =="

        if i >= 1:
            [sampleFile,number, setProbabilities] = generateSampleFileM(file, vectorNewProbabilities, "ADABOOST",i,expmnt)
            print sampleFile, number, setProbabilities, vectorNewProbabilities

        successful = False;

        while not successful:
            print "Numero Topicos, Rounds", numTopics, alpha, beta
            #numTopics = 3;


            print "Entrenando con Muestra";
            formatingFiles(numTopics,sampleFile,folder,alpha, beta)
            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,sampleFile,folder)
            [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)




            Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)

            #print "Distance_Emp"
            #print Distance_Emp

            confidenceAlpha  = float(0.5*math.log((1-Distance_Emp)/Distance_Emp));
            successful = True;
            confidenceParametersAlphas.append(confidenceAlpha);

            print "confidenceAlpha"
            print confidenceAlpha


        print "Probando con conjunto de Training"
        print "numTopics, alpha, beta"
        print numTopics, alpha, beta

        formatingFiles(numTopics,file,folder,alpha, beta)
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)


        while 0 in matrixTopWord.sum(axis=1):
            print "falla, probando de nuevo"
            formatingFiles(numTopics,file,folder,alpha, beta)
            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

        matrixWordTop = matrixTopWord.transpose()

        topicPriors = [float(1/float(numTopics))]*numTopics
        topicPriors = list(topicPriors)

        #print "topicPriors====", topicPriors
        matrixTopWord = generateInverseConditional(matrixWordTop,priorsCorpus, topicPriors)


        saveFileMatrix(matrixTopWord.transpose(),"MATRIXTOPWORDinv",i,file,"ADABOOST");
        saveFileMatrix(matrixTopWord,"MATRIXTOPWORD",i,file,"ADABOOST");
        saveFileMatrix(matrixDocTop,"MATRIXDOCTOP",i,file,"ADABOOST")
        saveFileMatrix(matrixProbTermsDoc,"MATRIXPROBTERMSDOCS",i,file,"ADABOOST")
        saveFileMatrix(matrixDocTermFinal,"MATRIXDOCTERMSFINAL",i,file,"ADABOOST")

        saveFileList(priorsCorpus,"PRIORSCORPUS",i,file,"ADABOOST")
        saveFileList(topicPriors,"TOPICPRIORS",i,file,"ADABOOST")



        [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
        #print "Perp Maquina: ", Perplexity_Prob
        machinesPerplexity.append(Perplexity_Emp);

        #print "Calculo Distancia"
        Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)

        vectorDifEmpProb  =[];
        vectorNewProbabilities = [];

        for l1, l2 in zip(probVectorEmp,probVectorProb):
            vectorDifEmpProb.append(float(l1-l2)/float(l1));

        for l1, l2 in zip(vectorDifEmpProb,setProbabilities):
            vectorNewProbabilities.append(float(l2)*math.exp(float(l1)*confidenceAlpha))


        sumProb = 0;

        for elem in vectorNewProbabilities:
            sumProb = sumProb + float(elem)

        vectorNewProbabilities[:] = [x/sumProb for x in vectorNewProbabilities]

        #print vectorNewProbabilities




        TOTAL_DTMO.append(matrixDocTerms)
        TOTAL_DTMProb.append(matrixProbTermsDoc)
        TOTAL_DTMEmp.append(matrixDocTermFinal)
        TOTAL_TOP_WORD_Emp.append(matrixTopWord.transpose());
        rows = matrixTopWord.transpose().shape[0]; cols = matrixTopWord.transpose().shape[1]
        TOTAL_TOP_PRIORS.append(numpy.array([matrixTopWord.sum(axis=1)])/rows);
        TOTAL_DOC_TOP_Emp.append(matrixDocTop);



        MEAN_TOTAL_DTMO = sum(TOTAL_DTMO)/(i+1)
        MEAN_TOTAL_DTMProb = sum(TOTAL_DTMProb)/(i+1)
        MEAN_TOTAL_DTMEmp = sum(TOTAL_DTMEmp)/(i+1)


        wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
        trainWords = [];
        for elem in wordsFileE: trainWords.append(elem.split(" ")[1])
        wordsFileE.close()


        trainPerp = trainPerplexityInv(trainWords, trainWords, testDocs, i, file,"ADABOOST")

        testPerp = trainPerplexityInv(trainWords, testWords, testDocs, i, file,"ADABOOST")

        [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(sum(TOTAL_DTMO)/(i+1),sum(TOTAL_DTMEmp)/(i+1),sum(TOTAL_DTMProb)/(i+1))

        Distance_Emp_Final = calculateDistance(sum(TOTAL_DTMO)/(i+1),sum(TOTAL_DTMEmp)/(i+1),sum(TOTAL_DTMProb)/(i+1))


        suma = 0;
        for l1, l2 in zip(confidenceParametersAlphas,TOTAL_DTMEmp):
            suma = suma + l1*l2
        #print suma

        suma =suma/numpy.array([suma.sum(axis=1)]).transpose()
        #print suma


        kld = KLD(MEAN_TOTAL_DTMEmp, MEAN_TOTAL_DTMProb)

        print "Resultado It, ", i
        print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld
        #return trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld

def BAGGING_MIXEDMODEL(listaTop, numWords, listaAlpha=[], listaBeta=[], itNumber=10,expmnt=1):

    import numpy as np
    method = "BAGGING_MIXEDMODEL"

    #========= SE INICIALIZAN LAS MATRICES DONDE SE GUARDARAN LOS RESULTADOS DE LOS ALGORITMOS ========#
    TOTAL_DTMTrivial = []; TOTAL_DTMO = []; TOTAL_DTMProb = []
    TOTAL_DTMEmp = []; TOTAL_TOP_WORD_Emp = []; TOTAL_TOP_PRIORS = [];
    TOTAL_DOC_TOP_Emp = [];

    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];

    import numpy

    #============ SE CALCULA EL NUMERO DE DOCUMENTOS EN EL DATASET ========#
    f = open("C:/Mallet/testCMD/"+file,"r")
    docs = 0;
    for line in f:
        docs = docs + 1;
    f.close();

    numDocs = docs

    prob = float(1/float(docs))

    #============ SE CALCULA LA DISTRIBUCION DE PROBABILIDAD DEL CONJUNTO  ========#
    setProbabilities = [];
    for doc in range(docs):
        setProbabilities.append(prob);

    folder = file.split("\\")[0];

    numTopics = listaTop[0]
    alpha = float(50/float(numTopics));
    beta = float(200/float(numWords));

    #============ CALCULO DE LOS PARAMETROS GENERALES DE CORPUS COMPLETO ========#
    formatingFiles(numTopics,file,folder,alpha, beta)
    [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

    priorsCorpusGral = priorsCorpus; priorsDocsGral = priorsDoc; priorsCorpusGral = priorsCorpus;
    matrixDocTermsGral = matrixDocTerms; matrixDocTermFinalGral = matrixDocTermFinal; matrixProbTermsDocGral = matrixProbTermsDoc;


    wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
    wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_gral.txt","w");


    #============ CALCULO DEL NUMERO DE PALABRAS ========#
    numTerms = 0;
    for elem in wordsFileA:
        wordsFileB.write(elem);
        numTerms = numTerms + 1;
    wordsFileA.close();
    wordsFileB.close();

    k = 0;

    topics = 0;
    topicsPerSubset = []

    iterations = itNumber;
    topicsTotal = 0;

    initTopics = numTopics;
    #============ ARCHIVO DONDE SE ESCRIBIRAN LOS RESULTADOS DEL EXPERIMENTO ========#
    expFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS"+str(expmnt)+"/Results.txt","w");
    for i in range(iterations):
        numTopics = initTopics
        [sampleFile,number, setProbabilities] = generateSampleFileM(file, setProbabilities, method, i, expmnt)
        print sampleFile, number, setProbabilities



        folder = file.split("\\")[0];

        time.sleep(0.2); random.seed(time.time());
        numTopics  = random.choice(listaTop);


        #============ SE CREAN LAS GRILLAS DE PARA LA CANTIDAD DE TOPICOS DE ENTRADA  ========#
        #============ Y SE ESCOGEN DOS PARAMETROS ALPHA Y BETA AL AZAR ========#
        alpha = float(50/float(numTopics))
        beta = float(200/float(numWords))
        time.sleep(0.2); random.seed(time.time()); numTopics  = random.choice(listaTop);
        alpha = float(50/float(numTopics)); listaAlpha = [alpha/100,alpha/10,alpha,alpha*10,alpha*100]
        beta = float(200/float(numWords)); listaBeta = [beta/100,beta/10,beta,beta*10,beta*100]
        time.sleep(0.2); random.seed(time.time()); alpha  = random.choice(listaAlpha);
        time.sleep(0.2); random.seed(time.time()); beta  = random.choice(listaBeta);

        print "n_topics","alpha","beta"
        print numTopics, alpha, beta


        #============ SE OBTIENE UN TOPIC MODEL CON EL CONJUNTO DE ENTRADA Y LOS PARAMETROS ESCOGIDOS ========#
        print "Probando con ", numTopics, alpha, beta
        formatingFiles(numTopics,sampleFile,folder,alpha, beta)
        wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
        wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_"+str(letters[k])+".txt","w");
        for elem in wordsFileA: #=====SE GUARDA CADA SALIDA DEL CONTEO DE PALABRAS EN ARCHIVOS DIFERENTES (words_count_file_*.txt)=====
            wordsFileB.write(elem);
        wordsFileA.close();
        wordsFileB.close();
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,sampleFile,folder)

        #============ SE COMPRUEBA DE QUE LA MATRIZ CONTENGA TODAS LAS FILAS SUMANDO 1 ========#
        while 0 in matrixTopWord.sum(axis=1):
            time.sleep(0.2); random.seed(time.time()); numTopics  = random.choice(listaTop);
            alpha = float(50/float(numTopics)); listaAlpha = [alpha/100,alpha/10,alpha,alpha*10,alpha*100]
            beta = float(200/float(numWords)); listaBeta = [beta/100,beta/10,beta,beta*10,beta*100]
            time.sleep(0.2); random.seed(time.time()); alpha  = random.choice(listaAlpha);
            time.sleep(0.2); random.seed(time.time()); beta  = random.choice(listaBeta);


            print "no resulto Probando con ", numTopics, alpha, beta
            formatingFiles(numTopics,sampleFile,folder,alpha, beta)
            wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
            wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_"+str(letters[k])+".txt","w");
            for elem in wordsFileA: #=====SE GUARDA CADA SALIDA DEL CONTEO DE PALABRAS EN ARCHIVOS DIFERENTES (words_count_file_*.txt)=====
                wordsFileB.write(elem);
            wordsFileA.close();
            wordsFileB.close();
            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,sampleFile,folder)



        topicsTotal = topicsTotal + numTopics;

        matrixWordTop = matrixTopWord.transpose()

        topicPriors = [float(1/float(numTopics))]*numTopics
        topicPriors = list(topicPriors)

        #============ MATRIZ DE LAS PROBABILIDADES DE LAS PALABRAS DADOS LOS TEMAS ========#
        matrixTopWord = generateInverseConditional(matrixWordTop,priorsCorpus, topicPriors)
        print matrixTopWord
        #print "topicPriors====", topicPriors


        rows = matrixWordTop.shape[0]; cols = matrixWordTop.shape[1]

        topics = topics + cols
        topicsPerSubset.append(cols)
        wordsFileC = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixTopWord_"+(letters[k])+".txt","w");
        for row in range(rows): #=====SE GUARDA CADA MATRIZ TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
            for col in range(cols):
                wordsFileC.write(str(matrixWordTop.item(row,col))+"\t")
            wordsFileC.write("\n");

        wordsFileC.close();

        rows = matrixDocTop.shape[0]; cols = matrixDocTop.shape[1]
        wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+(letters[k])+".txt","w");
        for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
            for col in range(cols):
                wordsFileD.write(str(matrixDocTop.item(row,col))+"\t")
            wordsFileD.write("\n");

        wordsFileD.close();

        TOTAL_DTMEmp.append(matrixDocTermFinal)
        TOTAL_TOP_WORD_Emp.append(matrixTopWord);
        rows = matrixTopWord.shape[0]; cols = matrixTopWord.shape[1]
        TOTAL_TOP_PRIORS.append(numpy.array([matrixWordTop.transpose().sum(axis=1)])/cols);
        TOTAL_DOC_TOP_Emp.append(matrixDocTop);

        #print "TOTAL_TOP_PRIORS"
        #print TOTAL_TOP_PRIORS
        sumTopPriors = 0;
        TOTAL_TOP_PRIORS_NORM = [];

        for elem in TOTAL_TOP_PRIORS:
            for num in elem[0]:
                sumTopPriors = sumTopPriors+ float(num);

        for elem in TOTAL_TOP_PRIORS:
            for num in elem[0]:
                TOTAL_TOP_PRIORS_NORM.append(float(num)/float(sumTopPriors))

        #print "TOTAL_TOP_PRIORS_NORM"
        #print TOTAL_TOP_PRIORS_NORM

        aggrMatrixTermTop = numpy.zeros((numTerms,topics))
        #print aggrMatrixTermTop

        print "PRIMER FOR ", "k ", k, "it ", i

        for let in range(k+1):
            #=====SE CONSIDERAN LOS ARCHIVOS DE LA ITERACION ACTUAL Y LOS DE LAS ITERACIONES ANTERIORES=====
            wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_"+(letters[let])+".txt","r");
            wordsFileC = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixTopWord_"+(letters[let])+".txt","r");
            wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/subSet_Words_Matrix_"+(letters[let])+".txt","w");


            for l1, l2 in zip(wordsFileB, wordsFileC):
                wordsFileD.write(str(l1).split(" ")[1]+"\t"+l2)
                #print l1, l2

            wordsFileB.close();
            wordsFileC.close();
            wordsFileD.close();



        #=====SE CREA LA MATRIZ "FUSIONADAN" DE TOPICO PALABRA A PARTIR DE LOS SUBCONJUNTOS
        wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_gral.txt","r");
        wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Words_Matrix_Gral.txt","w");
        wordsFileH = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_subSet_Words_Matrix_"+(letters[i])+".txt","w");

        savedString = ""

        allWordsString = "";

        #===== SE NORMALIZA EL VOCABULARIO DE LA MUESTRA PARA USANDO UNA MATRIZ CON ===
        #===== TODO EL VOCABULARIO DEL TRINING SET=====
        print "SEGUNDO FOR ", "k ", k, "it ", i
        words = 0;
        for elemE in wordsFileE:
            words = words + 1
            if words == int(0.25*numWords): print words, " de ", numWords
            if words == int(0.5*numWords): print words, " de ", numWords
            if words == int(0.75*numWords): print words, " de ", numWords



            savedString = "" #=====STRING DONDE SE CONCATENARA CADA FILA
            savedString = savedString + str(elemE).split(" ")[1]
            subSetSavedString = "";


            allWordsString = allWordsString+" "+str(elemE).split(" ")[1];


            for let, top in zip(range(k+1),topicsPerSubset):

                #============ ARCHIVO PROBABILIDADES DE CADA PALABRA EN CADA TOPICO POR CADA ITERACION  ========#
                wordsFileG = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/subSet_Words_Matrix_"+(letters[let])+".txt","r");

                flag = 0;
                for elemG in wordsFileG:

                    auxList = list(str(elemG).split("\t"));
                    if str(elemE).split(" ")[1] == str(auxList[0]): #=====SI LA PALABRA COINCIDE CON LA DEL SUBCONJUNTO
                        del auxList[-1]
                        del auxList[0]
                        savedString = savedString +" "+ ' '.join(auxList) #=====SE AGREGA LA FILA DE LA MATRIZ AL STRING

                        subSetSavedString = str(elemE).split(" ")[1] +" "+ ' '.join(auxList)
                        flag = 1;
                        break;
                if flag == 0:
                    for t in range(top):
                        savedString = savedString +" "+str(0.0);
                    subSetSavedString = str(elemE).split(" ")[1];
                    for t in range(top):
                         subSetSavedString = subSetSavedString +" "+str(0.0);


                wordsFileG.close();
            #print subSetSavedString
            #print savedString
            wordsFileH.write(subSetSavedString+"\n")
            wordsFileF.write(savedString+"\n")
        wordsFileE.close();
        wordsFileF.close();
        wordsFileH.close();



        #=====SE CREA LA MATRIZ NORMALIZADA PARA CADA UNO DE LOS SUBCONJUNTOS=====#

        wordsFileH = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_subSet_Words_Matrix_"+(letters[i])+".txt","r");

        SubSetNormMatrixTopTerm = numpy.zeros((numWords,numTopics))

        SubSetNormMatrixTermTop = numpy.zeros((numWords,numTopics))

        x = y = 0;
        for elemH in wordsFileH:
            elemH = str(elemH).replace("\n","")
            lst = str(elemH).split(" ")
            del lst[0]
            for el in lst:
                SubSetNormMatrixTermTop[x][y] = float(el)
                y = y + 1

            y = 0
            x = x  + 1;

        topicPriors = [float(1/float(numTopics))]*numTopics

        SubSetNormMatrixTopTerm = generateInverseConditional(SubSetNormMatrixTermTop,priorsCorpusGral, topicPriors)

        wordsFileH.close();

        wordsFileJ = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+(letters[i])+".txt","w");

        rows = SubSetNormMatrixTopTerm.shape[0]; cols = SubSetNormMatrixTopTerm.shape[1]
        for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
            for col in range(cols):
                wordsFileJ.write(str(SubSetNormMatrixTopTerm.item(row,col))+"\t")
            wordsFileJ.write("\n");
        wordsFileJ.close();


        rows = SubSetNormMatrixTopTerm.transpose().shape[0];
        cols = SubSetNormMatrixTopTerm.transpose().shape[1];

        f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+(letters[i])+"_Inv.txt","w");

        for fr in range(rows):
            for fc in range(cols):
                f.write(str(SubSetNormMatrixTopTerm.transpose()[fr][fc])+"\t")
            f.write("\n")
        f.close()


        #import sys;sys.exit("Error message")

        #print allWordsString

        #===== SE CREA LA MATRIZ FUSIONADA DE DOCUMENTO ToPICOS==
        aggrMatrixDocsTop = numpy.zeros((numDocs,topics))
        #print numDocs,topics
        #print aggrMatrixDocsTop

        print "TERCER FOR ", "k ", k, "it ", i
        top = 0;
        for let in range(k+1):
            doc = 0;
            print "C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+(letters[let])+".txt"
            wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+(letters[let])+".txt","r");
            for elem in wordsFileD:
                lst = str(elem).split("\t")
                #print lst
                del lst[-1]
                for count in range(len(lst)):
                    #print doc, top+count, count
                    aggrMatrixDocsTop[doc][top+count] = lst[count];
                doc = doc + 1;
            top = top + len(lst)
            wordsFileD.close();

        #print "MAT DOC-TOP"
        #pri
        # nt aggrMatrixDocsTop


        wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Document_Topic_Gral.txt","w");


        print "CUARTO FOR ", "k ", k, "it ", i
        rows = aggrMatrixDocsTop.shape[0]; cols = aggrMatrixDocsTop.shape[1]
        for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
            for col in range(cols):
                wordsFileF.write(str(aggrMatrixDocsTop.item(row,col))+"\t")
            wordsFileF.write("\n");
        wordsFileF.close();


        #===== PRIORS DE LOS TOPICOS Z =====
        zTopicsPrior = numpy.array([aggrMatrixDocsTop.sum(axis=0)])/aggrMatrixDocsTop.shape[0]

        #=====SE COPIA EL CONTENIDO DEL ARCHIVO "FUSIONADO" EN LA MATRIZ EN MEMORIA
        wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Words_Matrix_Gral.txt","r");
        aggrMatrixTermTop = numpy.zeros((numTerms,topics))

        i=0;
        print "QUINTO FOR ", "k ", k, "it ", i
        for elem in wordsFileF:
            lst = list(str(elem).split(" "))
            #print lst
            del lst[0];

            for j in range (topics+1):
                if len(lst) <= j:
                    break;
                else:
                    aggrMatrixTermTop[i][j] = float("{0:.10f}".format(float(lst[j])))

            i = i +1;


        #print aggrMatrixTermTop

        #=====SE NORMALIZA LA MATRIZ CREADA PARA QUE TENGA LA FORMA DE UNA PROBABILIDAD CONDICIONAL
        #print "MAT TERM-TOP"
        #print aggrMatrixTermTop
        aggrMatrixTermTop = aggrMatrixTermTop + 0.00000000000000001
        #print "MAT TERM-TOP"
        #print aggrMatrixTermTop


        aggrMatrixTermTopGral = aggrMatrixTermTop/numpy.array([aggrMatrixTermTop.sum(axis=1)]).transpose();



        #print "MAT TERM-TOP"
        #print aggrMatrixTermTopGral

        aggrMatrixTopTermGeneral = generateInverseConditional(aggrMatrixTermTopGral,priorsCorpusGral, TOTAL_TOP_PRIORS_NORM)

        #print "MAT TOP-TERM"
        #print aggrMatrixTopTermGeneral

        rows = aggrMatrixTermTopGral.shape[0]; cols = aggrMatrixTermTopGral.shape[1]

        wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_gral.txt","r");
        wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_gral_normalized.txt","w");
        savedString = "";

        i = 0;
        #=====SE GUARDA LA MATRIZ DE PROBABILIDADES CONDICIONALES=====
        wordList  =[];
        print "SEXTO FOR ", "k ", k, "it ", i
        for elem in wordsFileE:
            wordList.append(str(elem).split(" ")[1])

            savedString = savedString + str(elem).split(" ")[1]

            for j in range(cols):
                savedString = savedString + " " + str(float("{0:.3f}".format(float(aggrMatrixTermTopGral[i][j]))));
            i = i + 1;
            wordsFileF.write(savedString+"\n");
            savedString = "";

        wordsFileE.close();
        wordsFileF.close();


        aggrMatrixDocsTop = aggrMatrixDocsTop + 0.00000000000000001

        aggrMatrixDocsTopGral = aggrMatrixDocsTop/numpy.array([aggrMatrixDocsTop.sum(axis=1)]).transpose();

        print "MAT DOC-TOP"
        #print aggrMatrixDocsTopGral

        saveFileMatrix(aggrMatrixDocsTopGral,"MATRIXDOCTOP_MixedModel",0,file,"BAGGING_MIXEDMODEL");

        print "MAT TOP-TERM"
        #print aggrMatrixTopTermGeneral

        saveFileMatrix(aggrMatrixTopTermGeneral,"MATRIXTOPWORD_MixedModel",0,file,"BAGGING_MIXEDMODEL");
        saveFileMatrix(aggrMatrixTopTermGeneral.transpose(),"MATRIXTOPWORD_MixedModelinv",0,file,"BAGGING_MIXEDMODEL");


        wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
        trainWords = [];
        for elem in wordsFileE: trainWords.append(elem.split(" ")[1])
        wordsFileE.close()


        trainPerp = trainPerplexityInvMixedModel(Words_for_Train, Words_for_Train, testDocs, 0, file,"BAGGING_MIXEDMODEL")

        testPerp = trainPerplexityInvMixedModel(Words_for_Train, Words_for_Test, testDocs, 0, file,"BAGGING_MIXEDMODEL")


        aggrMatrixDocsTermGral =  np.dot(aggrMatrixDocsTopGral,aggrMatrixTopTermGeneral)

        print "MAT DOC-TERM"
        #print aggrMatrixDocsTermGral

        saveFileMatrix(aggrMatrixDocsTermGral,"MATRIXDOCTERMS_MixedModel",0,file,"BAGGING_MIXEDMODEL");

        #print numpy.array([aggrMatrixDocsTermGral.sum(axis=1)])

        aggrMatrixTermDocsGral = generateInverseConditional(aggrMatrixDocsTermGral,priorsDocsGral,priorsCorpusGral)
        #print aggrMatrixTermDocsGral
        #print matrixDocTermsGral
        #print matrixDocTermFinalGral
        #print matrixProbTermsDocGral

        [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(matrixDocTermsGral,aggrMatrixTermDocsGral,matrixProbTermsDocGral)
        print "Perplexity_Prob_Final, Perplexity_Emp_Final"
        print Perplexity_Prob_Final, Perplexity_Emp_Final


        Distance_Emp_Final = calculateDistance(matrixDocTermsGral,aggrMatrixTermDocsGral,matrixProbTermsDocGral)
        print "Distance_Emp_Final"
        print Distance_Emp_Final

        kld = KLD(aggrMatrixTermDocsGral,matrixProbTermsDocGral)

        print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld

        k = k + 1

        expFile.write(str(str(k)+"\t"+str(topicsTotal)+"\t"+str(numWords)+"\t"+str(alpha)+"\t"+str(beta)+"\t").replace('.',','))
        expFile.write(str(str(trainPerp)+"\t"+str(float(trainPerp)/float(topicsTotal))+"\t"+str(testPerp)+"\t"+str(float(testPerp)/float(topicsTotal))+"\t"+str(Perplexity_Emp_Final)+"\t"+str(Distance_Emp_Final)+"\t"+str(kld)+"\n").replace('.',','))

        #a = raw_input("next: ")

    expFile.close()


    print "iter ", itNumber

    for it in range(itNumber):
        for doc in Docs_for_Train_List:
            print it, doc
            trainPerp = trainPerplexityPerMachinePerDoc(Words_for_Train, doc, testDocs, it, file,"BAGGING_MIXEDMODEL")

    for it in range(itNumber):
        for doc in Docs_for_Test_List:
            print it, doc
            trainPerp = trainPerplexityPerMachinePerDoc(Words_for_Train, doc, testDocs, it, file,"BAGGING_MIXEDMODEL")


def ADABOOST_MIXEDMODEL(listaTop, numWords, listaAlpha=[], listaBeta=[], itNumber=10,expmnt=1):

    import numpy as np
    method = "ADABOOST_MIXEDMODEL"


    TOTAL_DTMTrivial = []; TOTAL_DTMO = []; TOTAL_DTMProb = []
    TOTAL_DTMEmp = []; TOTAL_TOP_WORD_Emp = []; TOTAL_TOP_PRIORS = [];
    TOTAL_DOC_TOP_Emp = [];

    #=====LISTA QUE PERMITE DIFERENCIAR ENTRE ITERACIONES=====

    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];

    import time
    import glob
    import numpy

    f = open("C:/Mallet/testCMD/"+file,"r")
    docs = 0;
    for line in f:
        docs = docs + 1;
    f.close();

    numDocs = docs

    prob = float(1/float(docs))

    setProbabilities = [];
    for doc in range(docs):
        setProbabilities.append(prob);

    [sampleFile,number, setProbabilities] = generateSampleFileM(file, setProbabilities, "ADABOOST_MIXEDMODEL", 0, expmnt)


    folder = file.split("\\")[0];

    numTopics = listaTop[0]
    alpha = float(50/float(numTopics));
    beta = float(200/float(numWords));

    formatingFiles(numTopics,file,folder,alpha, beta)
    [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

    priorsCorpusGral = priorsCorpus
    priorsDocsGral = priorsDoc
    matrixDocTermsGral = matrixDocTerms; matrixDocTermFinalGral = matrixDocTermFinal; matrixProbTermsDocGral = matrixProbTermsDoc;

    priorsCorpusGral = priorsCorpus;

    wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
    wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_gral.txt","w");

    numTerms = 0;
    for elem in wordsFileA:
        wordsFileB.write(elem);
        numTerms = numTerms + 1;
    wordsFileA.close();
    wordsFileB.close();

    k = 0;

    topics = 0;
    topicsPerSubset = []

    iterations = itNumber;
    topicsTotal = 0;

    initTopics = numTopics;
    expFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS"+str(expmnt)+"/Results.txt","w");

    machinesPerplexity = []; confidenceParametersAlphas = []; confidenceAlpha = 0;

    iterations = itNumber;


    for i in range(iterations):

        it = i

        print "==ITERACION: ", it, " =="

        if i >= 1:
            [sampleFile,number, setProbabilities] = generateSampleFileM(file, vectorNewProbabilities, "ADABOOST_MIXEDMODEL",i,expmnt)
            print sampleFile, number, setProbabilities, vectorNewProbabilities

        successful = False;

        while not successful:
            folder = file.split("\\")[0];

            time.sleep(0.2); random.seed(time.time());
            numTopics  = random.choice(listaTop);


            alpha = float(50/float(numTopics))
            beta = float(200/float(numWords))
            time.sleep(0.2); random.seed(time.time()); numTopics  = random.choice(listaTop);
            alpha = float(50/float(numTopics)); listaAlpha = [alpha/100,alpha/10,alpha,alpha*10,alpha*100]
            beta = float(200/float(numWords)); listaBeta = [beta/100,beta/10,beta,beta*10,beta*100]
            time.sleep(0.2); random.seed(time.time()); alpha  = random.choice(listaAlpha);
            time.sleep(0.2); random.seed(time.time()); beta  = random.choice(listaBeta);


            print "Numero Topicos, Rounds", numTopics, alpha, beta
            #numTopics = 3;



            print "Probando con ", numTopics, alpha, beta
            formatingFiles(numTopics,sampleFile,folder,alpha, beta)
            wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
            wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_"+str(letters[k])+".txt","w");
            for elem in wordsFileA: #=====SE GUARDA CADA SALIDA DEL CONTEO DE PALABRAS EN ARCHIVOS DIFERENTES (words_count_file_*.txt)=====
                wordsFileB.write(elem);
            wordsFileA.close();
            wordsFileB.close();
            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,sampleFile,folder)

            while 0 in matrixTopWord.sum(axis=1):
                time.sleep(0.2); random.seed(time.time()); numTopics  = random.choice(listaTop);
                alpha = float(50/float(numTopics)); listaAlpha = [alpha/100,alpha/10,alpha,alpha*10,alpha*100]
                beta = float(200/float(numWords)); listaBeta = [beta/100,beta/10,beta,beta*10,beta*100]
                time.sleep(0.2); random.seed(time.time()); alpha  = random.choice(listaAlpha);
                time.sleep(0.2); random.seed(time.time()); beta  = random.choice(listaBeta);


                print "no resulto Probando con ", numTopics, alpha, beta
                formatingFiles(numTopics,sampleFile,folder,alpha, beta)
                wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
                wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_"+str(letters[k])+".txt","w");
                for elem in wordsFileA: #=====SE GUARDA CADA SALIDA DEL CONTEO DE PALABRAS EN ARCHIVOS DIFERENTES (words_count_file_*.txt)=====
                    wordsFileB.write(elem);
                wordsFileA.close();
                wordsFileB.close();
                [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,sampleFile,folder)

            Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
            #===== CALCULO DEL PARAMETRO DE CONFIANZA DE LA ITERACION=====
            confidenceAlpha  = float(0.5*math.log((1-Distance_Emp)/Distance_Emp));
            successful = True;
            confidenceParametersAlphas.append(confidenceAlpha);

            print "confidenceAlpha"
            print confidenceAlpha


            topicsTotal = topicsTotal + numTopics;

            matrixWordTop = matrixTopWord.transpose()

            topicPriors = [float(1/float(numTopics))]*numTopics
            topicPriors = list(topicPriors)

            matrixTopWord = generateInverseConditional(matrixWordTop,priorsCorpus, topicPriors)
            print matrixTopWord
            print "topicPriors====", topicPriors

            rows = matrixWordTop.shape[0]; cols = matrixWordTop.shape[1]

            topics = topics + cols
            topicsPerSubset.append(cols)
            wordsFileC = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixTopWord_"+(letters[k])+".txt","w");
            for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
                for col in range(cols):
                    wordsFileC.write(str(matrixWordTop.item(row,col))+"\t")
                wordsFileC.write("\n");

            wordsFileC.close();

            rows = matrixDocTop.shape[0]; cols = matrixDocTop.shape[1]
            wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+(letters[k])+".txt","w");
            for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
                for col in range(cols):
                    wordsFileD.write(str(matrixDocTop.item(row,col))+"\t")
                wordsFileD.write("\n");

            wordsFileD.close();

            #TOTAL_DTMEmp.append(matrixDocTermFinal)
            TOTAL_TOP_WORD_Emp.append(matrixTopWord);
            rows = matrixTopWord.shape[0]; cols = matrixTopWord.shape[1]
            TOTAL_TOP_PRIORS.append(numpy.array([matrixWordTop.transpose().sum(axis=1)])/cols);
            TOTAL_DOC_TOP_Emp.append(matrixDocTop);

            print "TOTAL_TOP_PRIORS"
            print TOTAL_TOP_PRIORS
            sumTopPriors = 0;
            TOTAL_TOP_PRIORS_NORM = [];

            for elem in TOTAL_TOP_PRIORS:
                for num in elem[0]:
                    sumTopPriors = sumTopPriors+ float(num);

            for elem in TOTAL_TOP_PRIORS:
                for num in elem[0]:
                    TOTAL_TOP_PRIORS_NORM.append(float(num)/float(sumTopPriors))

            print "TOTAL_TOP_PRIORS_NORM"
            print TOTAL_TOP_PRIORS_NORM

            aggrMatrixTermTop = numpy.zeros((numTerms,topics))
            #print aggrMatrixTermTop

            print "PRIMER FOR ", "k ", k, "it ", i

            for let in range(k+1):
                wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_"+(letters[let])+".txt","r");
                wordsFileC = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixTopWord_"+(letters[let])+".txt","r");
                wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/subSet_Words_Matrix_"+(letters[let])+".txt","w");


                for l1, l2 in zip(wordsFileB, wordsFileC):
                    wordsFileD.write(str(l1).split(" ")[1]+"\t"+l2)
                    #print l1, l2

                wordsFileB.close();
                wordsFileC.close();
                wordsFileD.close();



            #=====SE CREA LA MATRIZ "FUSIONADAN" DE TOPICO PALABRA A PARTIR DE LOS SUBCONJUNTOS
            wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_gral.txt","r");
            wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Words_Matrix_Gral.txt","w");
            wordsFileH = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_subSet_Words_Matrix_"+(letters[i])+".txt","w");

            savedString = ""

            allWordsString = "";

            print "SEGUNDO FOR ", "k ", k, "it ", i
            words = 0;
            for elemE in wordsFileE:
                words = words + 1
                if words == int(0.25*numWords): print words, " de ", numWords
                if words == int(0.5*numWords): print words, " de ", numWords
                if words == int(0.75*numWords): print words, " de ", numWords



                savedString = "" #=====STRING DONDE SE CONCATENARA CADA FILA
                savedString = savedString + str(elemE).split(" ")[1]
                subSetSavedString = "";


                allWordsString = allWordsString+" "+str(elemE).split(" ")[1];


                for let, top in zip(range(k+1),topicsPerSubset):

                    #print "C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/subSet_Words_Matrix_"+(letters[let])+".txt"
                    wordsFileG = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/subSet_Words_Matrix_"+(letters[let])+".txt","r");

                    flag = 0;
                    for elemG in wordsFileG:

                        auxList = list(str(elemG).split("\t"));
                        if str(elemE).split(" ")[1] == str(auxList[0]): #=====SI LA PALABRA COINCIDE CON LA DEL SUBCONJUNTO
                            del auxList[-1]
                            del auxList[0]
                            savedString = savedString +" "+ ' '.join(auxList) #=====SE AGREGA LA FILA DE LA MATRIZ AL STRING

                            subSetSavedString = str(elemE).split(" ")[1] +" "+ ' '.join(auxList)
                            flag = 1;
                            break;
                    if flag == 0:
                        for t in range(top):
                            savedString = savedString +" "+str(0.0);
                        subSetSavedString = str(elemE).split(" ")[1];
                        for t in range(top):
                             subSetSavedString = subSetSavedString +" "+str(0.0);


                    wordsFileG.close();
                #print subSetSavedString
                #print savedString
                wordsFileH.write(subSetSavedString+"\n")
                wordsFileF.write(savedString+"\n")
            wordsFileE.close();
            wordsFileF.close();
            wordsFileH.close();



            #=====SE CREA LA MATRIZ NORMALIZADA PARA CADA UNO DE LOS SUBCONJUNTOS=====

            wordsFileH = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_subSet_Words_Matrix_"+(letters[i])+".txt","r");

            SubSetNormMatrixTopTerm = numpy.zeros((numWords,numTopics))

            SubSetNormMatrixTermTop = numpy.zeros((numWords,numTopics))

            x = y = 0;
            for elemH in wordsFileH:
                elemH = str(elemH).replace("\n","")
                lst = str(elemH).split(" ")
                del lst[0]
                for el in lst:
                    SubSetNormMatrixTermTop[x][y] = float(el)
                    y = y + 1

                y = 0
                x = x  + 1;

            topicPriors = [float(1/float(numTopics))]*numTopics

            SubSetNormMatrixTopTerm = generateInverseConditional(SubSetNormMatrixTermTop,priorsCorpusGral, topicPriors)

            wordsFileH.close();

            wordsFileJ = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+(letters[i])+".txt","w");

            rows = SubSetNormMatrixTopTerm.shape[0]; cols = SubSetNormMatrixTopTerm.shape[1]
            for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
                for col in range(cols):
                    wordsFileJ.write(str(SubSetNormMatrixTopTerm.item(row,col))+"\t")
                wordsFileJ.write("\n");
            wordsFileJ.close();


            rows = SubSetNormMatrixTopTerm.transpose().shape[0];
            cols = SubSetNormMatrixTopTerm.transpose().shape[1];

            f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+(letters[i])+"_Inv.txt","w");

            for fr in range(rows):
                for fc in range(cols):
                    f.write(str(SubSetNormMatrixTopTerm.transpose()[fr][fc])+"\t")
                f.write("\n")
            f.close()


            #import sys;sys.exit("Error message")

            #print allWordsString

            #===== SE CREA LA MATRIZ FUSIONADA DE DOCUMENTO ToPICOS==
            aggrMatrixDocsTop = numpy.zeros((numDocs,topics))
            #print numDocs,topics
            #print aggrMatrixDocsTop

            print "TERCER FOR ", "k ", k, "it ", i
            top = 0;
            for let in range(k+1):
                doc = 0;
                print "C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+(letters[let])+".txt"
                wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_"+(letters[let])+".txt","r");
                for elem in wordsFileD:
                    lst = str(elem).split("\t")
                    #print lst
                    del lst[-1]
                    for count in range(len(lst)):
                        #print doc, top+count, count
                        aggrMatrixDocsTop[doc][top+count] = lst[count];
                    doc = doc + 1;
                top = top + len(lst)
                wordsFileD.close();

            #print "MAT DOC-TOP"
            #pri
            # nt aggrMatrixDocsTop


            wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Document_Topic_Gral.txt","w");


            print "CUARTO FOR ", "k ", k, "it ", i
            rows = aggrMatrixDocsTop.shape[0]; cols = aggrMatrixDocsTop.shape[1]
            for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
                for col in range(cols):
                    wordsFileF.write(str(aggrMatrixDocsTop.item(row,col))+"\t")
                wordsFileF.write("\n");
            wordsFileF.close();


            #===== PRIORS DE LOS TOPICOS Z =====
            zTopicsPrior = numpy.array([aggrMatrixDocsTop.sum(axis=0)])/aggrMatrixDocsTop.shape[0]

            #=====SE COPIA EL CONTENIDO DEL ARCHIVO "FUSIONADO" EN LA MATRIZ EN MEMORIA
            wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Words_Matrix_Gral.txt","r");
            aggrMatrixTermTop = numpy.zeros((numTerms,topics))

            i=0;
            print "QUINTO FOR ", "k ", k, "it ", i
            for elem in wordsFileF:
                lst = list(str(elem).split(" "))
                #print lst
                del lst[0];

                for j in range (topics+1):
                    if len(lst) <= j:
                        break;
                    else:
                        aggrMatrixTermTop[i][j] = float("{0:.10f}".format(float(lst[j])))

                i = i +1;


            #print aggrMatrixTermTop

            #=====SE NORMALIZA LA MATRIZ CREADA PARA QUE TENGA LA FORMA DE UNA PROBABILIDAD CONDICIONAL
            #print "MAT TERM-TOP"
            #print aggrMatrixTermTop
            aggrMatrixTermTop = aggrMatrixTermTop + 0.00000000000000001
            #print "MAT TERM-TOP"
            #print aggrMatrixTermTop


            aggrMatrixTermTopGral = aggrMatrixTermTop/numpy.array([aggrMatrixTermTop.sum(axis=1)]).transpose();



            #print "MAT TERM-TOP"
            #print aggrMatrixTermTopGral

            aggrMatrixTopTermGeneral = generateInverseConditional(aggrMatrixTermTopGral,priorsCorpusGral, TOTAL_TOP_PRIORS_NORM)

            #print "MAT TOP-TERM"
            #print aggrMatrixTopTermGeneral

            rows = aggrMatrixTermTopGral.shape[0]; cols = aggrMatrixTermTopGral.shape[1]

            wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_gral.txt","r");
            wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_gral_normalized.txt","w");
            savedString = "";

            i = 0;
            #=====SE GUARDA LA MATRIZ DE PROBABILIDADES CONDICIONALES=====
            wordList  =[];
            print "SEXTO FOR ", "k ", k, "it ", i
            for elem in wordsFileE:
                wordList.append(str(elem).split(" ")[1])

                savedString = savedString + str(elem).split(" ")[1]

                for j in range(cols):
                    savedString = savedString + " " + str(float("{0:.3f}".format(float(aggrMatrixTermTopGral[i][j]))));
                i = i + 1;
                wordsFileF.write(savedString+"\n");
                savedString = "";

            wordsFileE.close();
            wordsFileF.close();


            aggrMatrixDocsTop = aggrMatrixDocsTop + 0.00000000000000001

            aggrMatrixDocsTopGral = aggrMatrixDocsTop/numpy.array([aggrMatrixDocsTop.sum(axis=1)]).transpose();

            print "MAT DOC-TOP"
            #print aggrMatrixDocsTopGral

            saveFileMatrix(aggrMatrixDocsTopGral,"MATRIXDOCTOP_MixedModel",0,file,"ADABOOST_MIXEDMODEL");

            print "MAT TOP-TERM"
            #print aggrMatrixTopTermGeneral

            saveFileMatrix(aggrMatrixTopTermGeneral,"MATRIXTOPWORD_MixedModel",0,file,"ADABOOST_MIXEDMODEL");
            saveFileMatrix(aggrMatrixTopTermGeneral.transpose(),"MATRIXTOPWORD_MixedModelinv",0,file,"ADABOOST_MIXEDMODEL");


            wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
            trainWords = [];
            for elem in wordsFileE: trainWords.append(elem.split(" ")[1])
            wordsFileE.close()


            trainPerp = trainPerplexityInvMixedModel(Words_for_Train, Words_for_Train, testDocs, 0, file,"ADABOOST_MIXEDMODEL")

            testPerp = trainPerplexityInvMixedModel(Words_for_Train, Words_for_Test, testDocs, 0, file,"ADABOOST_MIXEDMODEL")


            aggrMatrixDocsTermGral =  np.dot(aggrMatrixDocsTopGral,aggrMatrixTopTermGeneral)

            print "MAT DOC-TERM"
            #print aggrMatrixDocsTermGral

            saveFileMatrix(aggrMatrixDocsTermGral,"MATRIXDOCTERMS_MixedModel",0,file,"ADABOOST_MIXEDMODEL");

            #print numpy.array([aggrMatrixDocsTermGral.sum(axis=1)])

            aggrMatrixTermDocsGral = generateInverseConditional(aggrMatrixDocsTermGral,priorsDocsGral,priorsCorpusGral)
            #print aggrMatrixTermDocsGral
            #print matrixDocTermsGral
            #print matrixDocTermFinalGral
            #print matrixProbTermsDocGral

            [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(matrixDocTermsGral,aggrMatrixTermDocsGral,matrixProbTermsDocGral)
            print "Perplexity_Prob_Final, Perplexity_Emp_Final"
            print Perplexity_Prob_Final, Perplexity_Emp_Final


            Distance_Emp_Final = calculateDistance(matrixDocTermsGral,aggrMatrixTermDocsGral,matrixProbTermsDocGral)
            print "Distance_Emp_Final"
            print Distance_Emp_Final

            kld = KLD(aggrMatrixTermDocsGral,matrixProbTermsDocGral)

            print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld

            k = k + 1

            expFile.write(str(str(k)+"\t"+str(topicsTotal)+"\t"+str(numWords)+"\t"+str(alpha)+"\t"+str(beta)+"\t").replace('.',','))
            expFile.write(str(str(trainPerp)+"\t"+str(float(trainPerp)/float(topicsTotal))+"\t"+str(testPerp)+"\t"+str(float(testPerp)/float(topicsTotal))+"\t"+str(Perplexity_Emp_Final)+"\t"+str(Distance_Emp_Final)+"\t"+str(kld)+"\n").replace('.',','))

            #a = raw_input("next: ")









        print "Probando con conjunto de Training"
        print "numTopics, alpha, beta"
        print numTopics, alpha, beta

        formatingFiles(numTopics,file,folder,alpha, beta)
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)


        while 0 in matrixTopWord.sum(axis=1):
            print "falla, probando de nuevo"
            formatingFiles(numTopics,file,folder,alpha, beta)
            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

        matrixWordTop = matrixTopWord.transpose()

        topicPriors = [float(1/float(numTopics))]*numTopics
        topicPriors = list(topicPriors)

        #print "topicPriors====", topicPriors
        matrixTopWord = generateInverseConditional(matrixWordTop,priorsCorpus, topicPriors)


        print "i ", i

        saveFileMatrix(matrixTopWord.transpose(),"MATRIXTOPWORDinv",it,file,"ADABOOST_MIXEDMODEL");
        saveFileMatrix(matrixTopWord,"MATRIXTOPWORD",it,file,"ADABOOST_MIXEDMODEL");
        saveFileMatrix(matrixDocTop,"MATRIXDOCTOP",it,file,"ADABOOST_MIXEDMODEL")
        saveFileMatrix(matrixProbTermsDoc,"MATRIXPROBTERMSDOCS",it,file,"ADABOOST_MIXEDMODEL")
        saveFileMatrix(matrixDocTermFinal,"MATRIXDOCTERMSFINAL",it,file,"ADABOOST_MIXEDMODEL")

        saveFileList(priorsCorpus,"PRIORSCORPUS",it,file,"ADABOOST_MIXEDMODEL")
        saveFileList(topicPriors,"TOPICPRIORS",it,file,"ADABOOST_MIXEDMODEL")



        [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
        #print "Perp Maquina: ", Perplexity_Prob
        machinesPerplexity.append(Perplexity_Emp);

        #print "Calculo Distancia"
        Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)

        vectorDifEmpProb  =[];
        vectorNewProbabilities = [];

        for l1, l2 in zip(probVectorEmp,probVectorProb):
            vectorDifEmpProb.append(float(l1-l2)/float(l1));


        for l1, l2 in zip(vectorDifEmpProb,setProbabilities):
            vectorNewProbabilities.append(float(l2)*math.exp(float(l1)*confidenceAlpha))


        sumProb = 0;

        for elem in vectorNewProbabilities:
            sumProb = sumProb + float(elem)

        vectorNewProbabilities[:] = [x/sumProb for x in vectorNewProbabilities]

        #print vectorNewProbabilities




        TOTAL_DTMO.append(matrixDocTerms)
        TOTAL_DTMProb.append(matrixProbTermsDoc)
        TOTAL_DTMEmp.append(matrixDocTermFinal)
        TOTAL_TOP_WORD_Emp.append(matrixTopWord.transpose());
        rows = matrixTopWord.transpose().shape[0]; cols = matrixTopWord.transpose().shape[1]
        #TOTAL_TOP_PRIORS.append(numpy.array([matrixTopWord.sum(axis=1)])/rows);
        TOTAL_DOC_TOP_Emp.append(matrixDocTop);


        print matrixDocTerms.shape[0],matrixDocTerms.shape[1], len(TOTAL_DTMO)
        print matrixProbTermsDoc.shape[0],matrixProbTermsDoc.shape[1], len(TOTAL_DTMProb)
        print matrixDocTermFinal.shape[0],matrixDocTermFinal.shape[1], len(TOTAL_DTMEmp)

        MEAN_TOTAL_DTMO = sum(TOTAL_DTMO)/(it+1)
        MEAN_TOTAL_DTMProb = sum(TOTAL_DTMProb)/(it+1)
        MEAN_TOTAL_DTMEmp = sum(TOTAL_DTMEmp)/(it+1)


        wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
        trainWords = [];
        for elem in wordsFileE: trainWords.append(elem.split(" ")[1])
        wordsFileE.close()


        trainPerp = trainPerplexityInv(trainWords, trainWords, testDocs, it, file,"ADABOOST_MIXEDMODEL")

        testPerp = trainPerplexityInv(trainWords, testWords, testDocs, it, file,"ADABOOST_MIXEDMODEL")

        [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(sum(TOTAL_DTMO)/(it+1),sum(TOTAL_DTMEmp)/(it+1),sum(TOTAL_DTMProb)/(it+1))

        Distance_Emp_Final = calculateDistance(sum(TOTAL_DTMO)/(it+1),sum(TOTAL_DTMEmp)/(it+1),sum(TOTAL_DTMProb)/(it+1))


        suma = 0;
        for l1, l2 in zip(confidenceParametersAlphas,TOTAL_DTMEmp):
            suma = suma + l1*l2
        #print suma

        suma =suma/numpy.array([suma.sum(axis=1)]).transpose()
        #print suma


        kld = KLD(MEAN_TOTAL_DTMEmp, MEAN_TOTAL_DTMProb)

        print "Resultado It, ", it
        print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld
        #return trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld

    expFile.close()
    print "aca2"
    import sys; sys.exit("Error message")


def trainPerplexityPerMachinePerDoc(trainWords, testWords, testDocs, iterations, file,method):

    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];

    print "letter ", str(letters[iterations])

    numTopics = 0;
    f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+str(letters[iterations])+".txt","r");
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

    f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+str(letters[iterations])+"_Inv.txt","r");
    linea = 0;
    for line2 in f:
        listaPalabra = [];
        if linea in testWordsIndex:

            listaPalabra = str(line2).split("\t"); del listaPalabra[-1]

            listaPalabra = [float(elem)+0.0000000000000001 for elem in listaPalabra]
            #print listaPalabra
            listaPalabra = [math.log(float(elem)*topicPriors) for elem in listaPalabra]
            sumaPerp =  sum(listaPalabra) + sumaPerp

        linea = linea + 1;
    f.close()

    #print "sumaPerp ", sumaPerp

    #print "==Comienza Ultima=="

    f = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/Normal_SubSet_Top_Words_Matrix_"+str(letters[iterations])+".txt","r");
    linea = 0;
    for line2 in f:
        listaPalabra = [];
        listaPalabra = str(line2).split("\t"); del listaPalabra[-1]
        #print listaPalabra
        listaPalabra =  [listaPalabra[x] for x in testWordsIndex]
        #print listaPalabra
        listaPalabra = [float(elem)+0.0000000000000001 for elem in listaPalabra]
        #print listaPalabra
        listaPalabra = [math.log(float(elem)*topicPriors) for elem in listaPalabra]
        #print listaPalabra
        print sum(listaPalabra)


    f.close()






TOTAL_DTMTrivial = []; TOTAL_DTMO = []; TOTAL_DTMProb = []
TOTAL_DTMEmp = []; TOTAL_TOP_WORD_Emp = []; TOTAL_TOP_PRIORS = [];
TOTAL_DOC_TOP_Emp = [];



dataset = "Dataset4";
#method = "LDA"
#method = "BAGGING_MIXEDMODEL"
method = "ADABOOST_MIXEDMODEL"


print "Contar Topicos Si:1, No:2 :"
contarTopicos = int(raw_input("Seleccione: "))

print "Calcular Salidas? Si:1, No:2 :"
promediar = int(raw_input("Seleccione: "))



#2 temas max
if dataset == "DatasetONG":
    nombreData = "DatasetONG"
    file = "DS_ONG_FRMT\\Z_ONG_General_TRAIN.txt";
    testfile = "DS_ONG_FRMT\\Z_ONG_General_TEST.txt";
    numberOfTopics = [2];


#2 temas max
if dataset == "Dataset4":
    nombreData = "Dataset4Train"
    file = "Datastet_LDA_Toy_Cuatro\\TrainWordsPresentationExample.txt";
    testfile = "Datastet_LDA_Toy_Cuatro\\TrainWordsPresentationExampleTest.txt";
    numberOfTopics = [2];

#100 temas max
if dataset == "Dataset5":
    nombreData = "50DocsTrain"
    file = "trainDOS\\50DocsTrain.txt";
    testfile = "trainDOS\\50DocsTest.txt";
    numberOfTopics = [2];

#if dataset == "Dataset6":
#    nombreData = "Dataset6_CorpusCompleto"
#    file = "DatasetCompleto\\corpusTrain.txt";
#    testfile = "DatasetCompleto\\corpusTest.txt";
#    numberOfTopics = [10];

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


#if dataset == "Dataset6": file = "trainUNO\\corpus.txt"; testfile = "trainUNO\\corpusTest.txt";

#if dataset == "Dataset7": file = "trainTRES\\SampleTrainCorpus.txt"; testfile = "trainTRES\\SampleTestCorpus.txt";

#perpFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/PerplexityResume.txt","w");
#distFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/DistanceResume.txt","w");
dataSetProbFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/DatasetProbabilities.txt","w");


f = open("C:/Mallet/testCMD/"+file,"r"); cont = 0;
for elem in f: cont = cont + 1;
f.close();
f = open("C:/Mallet/testCMD/"+file,"r");
for elem in f: dataSetProbFile.write(str(float(1/float(cont)))+"\n");
f.close();
dataSetProbFile.close();


#===========OBTENER PALABRAS DE ENTRENAMIENTO=================
numTopics = 3
alpha = 10;
beta = 0.1;

folder = file.split("\\")[0];
print folder, file
formatingFiles(numTopics,file,folder,alpha, beta)
[matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)


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

print trainDocs
print trainWords
print testDocs
print testWords


#import sys; sys.exit("Error message")

if contarTopicos == 1:
    print "nombreData: ", nombreData
    print "Probando con ", numTopics, alpha, beta
    formatingFiles(numTopics,file,folder,alpha, beta)
    [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

    resta = int(numTopics*0.10)
    while 0 in matrixTopWord.sum(axis=1):
        numTopics = int(numTopics - resta)
        print "No resulta, probando con ", numTopics, " topicos"
        formatingFiles(numTopics,file,folder,alpha, beta)
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

    import sys; sys.exit("Error message")



print numTopics, numWords, numTestWords ,alpha, beta

if method == "ADABOOST_MIXEDMODEL":
    time.sleep(0.2)
    random.seed(time.time());
    numTopics  = random.choice(numberOfTopics);


    if dataset == "Dataset4": listaTop = [2,3]
    if dataset == "Dataset5": listaTop = [10,15,20,25,30]
    #if dataset == "Dataset6": listaTop = [20]
    if dataset == "Dataset6_talk_religion_misc": listaTop = [16,32,48,64,80]
    if dataset == "Dataset6_talk_politics_misc": listaTop = [16,32,48,64,80]
    if dataset == "Dataset6_talk_politics_mideast": listaTop = [24,48,72,96,120]
    if dataset == "Dataset6_talk_politics_guns": listaTop = [20,40,60,80,100]
    if dataset == "Dataset6_soc_religion_christian": listaTop = [28,56,84,112,140]
    if dataset == "Dataset6_sci_space": listaTop = [24,48,72,96,120]
    if dataset == "Dataset6_sci_med": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_sci_electronics": listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_sci_crypt": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_rec_sport_hockey": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_rec_sport_baseball": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_rec_motorcycles": listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_rec_autos": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_misc_forsale":  listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_comp_windows_x":  listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_comp_sys_mac_hardware":  listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_comp_sys_ibm_pc_hardware":  listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_comp_os_ms_windows_misc":  listaTop = [14,28,42,56,70]
    if dataset == "Dataset6_comp_graphics":  listaTop = [16,32,48,64,80]
    if dataset == "Dataset6_alt_atheism":  listaTop = [20,40,60,80,100]



    import os
    import glob

    print "ENTRA FUNC ADABOOST"
        #=====SE ELIMINAN LOS ARCHIVOS CREADOS ANTERIORMENTE====
    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/SubSetFile*.txt")

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixTopWord_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/subSet_Words_Matrix_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS1/*")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS2/*")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS3/*")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS4/*")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS5/*")
    for subfile in files: os.remove(subfile)


    expNumber = [1]
    print "FIN"



    for exp in expNumber:
        listaAlpha=[]
        listaBeta=[]
        ADABOOST_MIXEDMODEL(listaTop, numWords, listaAlpha, listaBeta,10,exp)

    import sys; sys.exit("Error message")

print numTopics, numWords, numTestWords ,alpha, beta

if method == "BAGGING_MIXEDMODEL":
    time.sleep(0.2)
    random.seed(time.time());
    numTopics  = random.choice(numberOfTopics);


    if dataset == "Dataset4": listaTop = [2,3]
    if dataset == "Dataset5": listaTop = [10,15,20,25,30]
    #if dataset == "Dataset6": listaTop = [20]
    if dataset == "Dataset6_talk_religion_misc": listaTop = [16,32,48,64,80]
    if dataset == "Dataset6_talk_politics_misc": listaTop = [16,32,48,64,80]
    if dataset == "Dataset6_talk_politics_mideast": listaTop = [24,48,72,96,120]
    if dataset == "Dataset6_talk_politics_guns": listaTop = [20,40,60,80,100]
    if dataset == "Dataset6_soc_religion_christian": listaTop = [28,56,84,112,140]
    if dataset == "Dataset6_sci_space": listaTop = [24,48,72,96,120]
    if dataset == "Dataset6_sci_med": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_sci_electronics": listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_sci_crypt": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_rec_sport_hockey": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_rec_sport_baseball": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_rec_motorcycles": listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_rec_autos": listaTop = [18,36,54,72,90]
    if dataset == "Dataset6_misc_forsale":  listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_comp_windows_x":  listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_comp_sys_mac_hardware":  listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_comp_sys_ibm_pc_hardware":  listaTop = [12,24,36,48,60]
    if dataset == "Dataset6_comp_os_ms_windows_misc":  listaTop = [14,28,42,56,70]
    if dataset == "Dataset6_comp_graphics":  listaTop = [16,32,48,64,80]
    if dataset == "Dataset6_alt_atheism":  listaTop = [20,40,60,80,100]



    import os
    import glob

    print "ENTRA FUNC BAGGING"
        #=====SE ELIMINAN LOS ARCHIVOS CREADOS ANTERIORMENTE====
    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/SubSetFile*.txt")

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/words_count_file_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixTopWord_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/subSet_Words_Matrix_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/matrixDocTop_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS1/*")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS2/*")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS3/*")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS4/*")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/"+method+"/RESULTS5/*")
    for subfile in files: os.remove(subfile)


    expNumber = [1]

    for exp in expNumber:
        listaAlpha=[]
        listaBeta=[]
        BAGGING_MIXEDMODEL(listaTop, numWords, listaAlpha, listaBeta,10,exp)

    import sys; sys.exit("Error message")

if method == "LDA":

    if dataset == "DatasetONG":
        numTopics = 3
        print numTopics, numWords, alpha, beta
    print "ENTRA FUNC LDA con", numTopics, numWords, alpha, beta
    #import sys; sys.exit("Error message")
    LDA(numTopics, numWords, alpha, beta)


if method == "LDARep":

    import os
    import glob

    expNumber = [1]

    if dataset == "DatasetONG": listaTop = [3]
    #if dataset == "Dataset4": listaTop = [2]
    #if dataset == "Dataset5": listaTop = [2,4,6,8,9]
    #if dataset == "Dataset6": listaTop = [20]
    #if dataset == "Dataset6_talk_religion_misc": listaTop = [16,32,48,64,80]
    #if dataset == "Dataset6_talk_politics_misc": listaTop = [16,32,48,64,80]
    #if dataset == "Dataset6_talk_politics_mideast": listaTop = [24,48,72,96,120]
    #if dataset == "Dataset6_talk_politics_guns": listaTop = [20,40,60,80,100]
    #if dataset == "Dataset6_soc_religion_christian": listaTop = [28,56,84,112,140]
    #if dataset == "Dataset6_sci_space": listaTop = [24,48,72,96,120]
    #if dataset == "Dataset6_sci_med": listaTop = [18,36,54,72,90]
    #if dataset == "Dataset6_sci_electronics": listaTop = [12,24,36,48,60]
    #if dataset == "Dataset6_sci_crypt": listaTop = [18,36,54,72,90]
    #if dataset == "Dataset6_rec_sport_hockey": listaTop = [18,36,54,72,90]
    #if dataset == "Dataset6_rec_sport_baseball": listaTop = [18,36,54,72,90]
    #if dataset == "Dataset6_rec_motorcycles": listaTop = [12,24,36,48,60]
    #if dataset == "Dataset6_rec_autos": listaTop = [18,36,54,72,90]
    #if dataset == "Dataset6_misc_forsale":  listaTop = [12,24,36,48,60]
    #if dataset == "Dataset6_comp_windows_x":  listaTop = [12,24,36,48,60]
    #if dataset == "Dataset6_comp_sys_mac_hardware":  listaTop = [12,24,36,48,60]
    #if dataset == "Dataset6_comp_sys_ibm_pc_hardware":  listaTop = [12,24,36,48,60]
    #if dataset == "Dataset6_comp_os_ms_windows_misc":  listaTop = [14,28,42,56,70]
    #if dataset == "Dataset6_comp_graphics":  listaTop = [16,32,48,64,80]
    #if dataset == "Dataset6_alt_atheism":  listaTop = [20,40,60,80,100]

    #listaAlpha = [float(alpha/100),float(alpha/10), alpha, float(alpha*10), float(alpha*100)]
    #listaBeta = [float(beta/10), beta, float(beta*10)]
    #listaAlpha = [alpha]
    #listaBeta = [beta]
    if promediar == 3:

        for top in listaTop:
            listaPLLTrain = []
            listaPLLTest = []
            listaPerplexity = []
            listaDistance = []
            listaKLD = []
            for exp in expNumber:

                PhiLLTrainMatrix  = numpy.zeros((5,5))
                PhiLLTestMatrix  = numpy.zeros((5,5))
                PerplexityMatrix  = numpy.zeros((5,5))
                DistanceMatrix  = numpy.zeros((5,5))
                KLDMatrix  = numpy.zeros((5,5))
                expFileTest = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTest_top"+str(top)+".txt","r");
                FilePLLTrain = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTrain_top"+str(top)+".txt","r");
                FilePLLTest = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTest_top"+str(top)+".txt","r");
                FilePerplexity = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Perplexity_top"+str(top)+".txt","r");
                FileDistance = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Distance_top"+str(top)+".txt","r");
                FileKLD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/KLD_top"+str(top)+".txt","r");

                x=0;y=0;
                for line,line2,line3,line4,line5, in zip(FilePLLTrain,FilePLLTest,FilePerplexity,FileDistance,FileKLD):
                    for l1,l2,l3,l4,l5 in zip(line.strip().split("\t"),line2.strip().split("\t"),line3.strip().split("\t"),line4.strip().split("\t"),line5.strip().split("\t")):
                        PhiLLTrainMatrix[x][y] = float(str(l1).replace(",","."));
                        PhiLLTestMatrix[x][y] = float(str(l2).replace(",","."));
                        PerplexityMatrix[x][y] = float(str(l3).replace(",","."));
                        DistanceMatrix[x][y] = float(str(l4).replace(",","."));
                        KLDMatrix[x][y] = float(str(l5).replace(",","."));
                        y= y+1;
                    y=0;
                    x=x+1;
                FilePLLTrain.close();
                FilePLLTest.close();
                FilePerplexity.close();
                FileDistance.close();
                FileKLD.close();

                listaPLLTrain.append(PhiLLTrainMatrix)
                listaPLLTest.append(PhiLLTestMatrix)
                listaPerplexity.append(PerplexityMatrix)
                listaDistance.append(DistanceMatrix)
                listaKLD.append(KLDMatrix)

            meanMatrixPLLTrain  = numpy.zeros((5,5))
            sdtMatrixPLLTrain = numpy.zeros((5,5))
            for x in range(5):
                for y in range(5):
                    #print  float(listaPLLTrain[0][x][y]),float(listaPLLTrain[1][x][y]), float(listaPLLTrain[2][x][y]), float(listaPLLTrain[3][x][y]), float(listaPLLTrain[4][x][y])
                    if float(listaPLLTrain[0][x][y]) != 0.0:
                        if float(listaPLLTrain[1][x][y]) != 0.0:
                            if float(listaPLLTrain[2][x][y]) != 0.0:
                                if float(listaPLLTrain[3][x][y]) != 0.0:
                                    if float(listaPLLTrain[4][x][y]) != 0.0:
                                        meanMatrixPLLTrain[x][y] = numpy.mean([float(listaPLLTrain[0][x][y]),float(listaPLLTrain[1][x][y]),float(listaPLLTrain[2][x][y]),float(listaPLLTrain[3][x][y]),float(listaPLLTrain[4][x][y])])
                                        sdtMatrixPLLTrain[x][y] = numpy.std([float(listaPLLTrain[0][x][y]),float(listaPLLTrain[1][x][y]),float(listaPLLTrain[2][x][y]),float(listaPLLTrain[3][x][y]),float(listaPLLTrain[4][x][y])])

            print meanMatrixPLLTrain
            print sdtMatrixPLLTrain


            meanMatrixPLLTest  = numpy.zeros((5,5))
            sdtMatrixPLLTest = numpy.zeros((5,5))
            for x in range(5):
                for y in range(5):
                    #print  float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]), float(listaPLLTest[2][x][y]), float(listaPLLTest[3][x][y]), float(listaPLLTest[4][x][y])
                    if float(listaPLLTest[0][x][y]) != 0.0:
                        if float(listaPLLTest[1][x][y]) != 0.0:
                            if float(listaPLLTest[2][x][y]) != 0.0:
                                if float(listaPLLTest[3][x][y]) != 0.0:
                                    if float(listaPLLTest[4][x][y]) != 0.0:
                                        meanMatrixPLLTest[x][y] = numpy.mean([float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]),float(listaPLLTest[2][x][y]),float(listaPLLTest[3][x][y]),float(listaPLLTest[4][x][y])])
                                        sdtMatrixPLLTest[x][y] = numpy.std([float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]),float(listaPLLTest[2][x][y]),float(listaPLLTest[3][x][y]),float(listaPLLTest[4][x][y])])

            print meanMatrixPLLTest
            print sdtMatrixPLLTest

            meanMatrixPerplexity  = numpy.zeros((5,5))
            sdtMatrixPerplexity = numpy.zeros((5,5))
            for x in range(5):
                for y in range(5):
                    #print  float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]), float(listaPLLTest[2][x][y]), float(listaPLLTest[3][x][y]), float(listaPLLTest[4][x][y])
                    if float(listaPerplexity[0][x][y]) != 0.0:
                        if float(listaPerplexity[1][x][y]) != 0.0:
                            if float(listaPerplexity[2][x][y]) != 0.0:
                                if float(listaPerplexity[3][x][y]) != 0.0:
                                    if float(listaPerplexity[4][x][y]) != 0.0:
                                        meanMatrixPerplexity[x][y] = numpy.mean([float(listaPerplexity[0][x][y]),float(listaPerplexity[1][x][y]),float(listaPerplexity[2][x][y]),float(listaPerplexity[3][x][y]),float(listaPerplexity[4][x][y])])
                                        sdtMatrixPerplexity[x][y] = numpy.std([float(listaPerplexity[0][x][y]),float(listaPerplexity[1][x][y]),float(listaPerplexity[2][x][y]),float(listaPerplexity[3][x][y]),float(listaPerplexity[4][x][y])])

            print meanMatrixPerplexity
            print sdtMatrixPerplexity


            meanMatrixDistance  = numpy.zeros((5,5))
            sdtMatrixDistance = numpy.zeros((5,5))
            for x in range(5):
                for y in range(5):
                    #print  float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]), float(listaPLLTest[2][x][y]), float(listaPLLTest[3][x][y]), float(listaPLLTest[4][x][y])
                    if float(listaDistance[0][x][y]) != 0.0:
                        if float(listaDistance[1][x][y]) != 0.0:
                            if float(listaDistance[2][x][y]) != 0.0:
                                if float(listaDistance[3][x][y]) != 0.0:
                                    if float(listaDistance[4][x][y]) != 0.0:
                                        meanMatrixDistance[x][y] = numpy.mean([float(listaDistance[0][x][y]),float(listaDistance[1][x][y]),float(listaDistance[2][x][y]),float(listaDistance[3][x][y]),float(listaDistance[4][x][y])])
                                        sdtMatrixDistance[x][y] = numpy.std([float(listaDistance[0][x][y]),float(listaDistance[1][x][y]),float(listaDistance[2][x][y]),float(listaDistance[3][x][y]),float(listaDistance[4][x][y])])

            print meanMatrixDistance
            print sdtMatrixDistance

            meanMatrixKLD  = numpy.zeros((5,5))
            sdtMatrixKLD = numpy.zeros((5,5))
            for x in range(5):
                for y in range(5):
                    #print  float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]), float(listaPLLTest[2][x][y]), float(listaPLLTest[3][x][y]), float(listaPLLTest[4][x][y])
                    if float(listaKLD[0][x][y]) != 0.0:
                        if float(listaKLD[1][x][y]) != 0.0:
                            if float(listaKLD[2][x][y]) != 0.0:
                                if float(listaKLD[3][x][y]) != 0.0:
                                    if float(listaKLD[4][x][y]) != 0.0:
                                        meanMatrixKLD[x][y] = numpy.mean([float(listaKLD[0][x][y]),float(listaKLD[1][x][y]),float(listaKLD[2][x][y]),float(listaKLD[3][x][y]),float(listaKLD[4][x][y])])
                                        sdtMatrixKLD[x][y] = numpy.std([float(listaKLD[0][x][y]),float(listaKLD[1][x][y]),float(listaKLD[2][x][y]),float(listaKLD[3][x][y]),float(listaKLD[4][x][y])])

            print meanMatrixKLD
            print sdtMatrixKLD

            FilePLLTrainMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PHLLTrainMean_"+str(top)+".txt","w");
            FilePLLTrainSTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PHLLTrainSTD_"+str(top)+".txt","w");
            FilePLLTestMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PHLLTestMean_"+str(top)+".txt","w");
            FilePLLTestSTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PHLLTestSTD_"+str(top)+".txt","w");
            FilePerplexityMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PerplexityMean_"+str(top)+".txt","w");
            FilePerplexitySTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PerplexitySTD_"+str(top)+".txt","w");
            FileDistanceMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/DistanceMean_"+str(top)+".txt","w");
            FileDistanceSTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/DistanceSTD_"+str(top)+".txt","w");
            FileKLDMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/KLDMean_"+str(top)+".txt","w");
            FileKLDSTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/KLDSTD_"+str(top)+".txt","w");


            for x in range (5):
                for y in range(5):
                    FilePLLTrainMean.write(str(meanMatrixPLLTrain.item(x,y)).replace('.',',')+"\t")
                    FilePLLTrainSTD.write(str(sdtMatrixPLLTrain.item(x,y)).replace('.',',')+"\t")
                    FilePLLTestMean.write(str(meanMatrixPLLTest.item(x,y)).replace('.',',')+"\t")
                    FilePLLTestSTD.write(str(sdtMatrixPLLTest.item(x,y)).replace('.',',')+"\t")
                    FilePerplexityMean.write(str(meanMatrixPerplexity.item(x,y)).replace('.',',')+"\t")
                    FilePerplexitySTD.write(str(sdtMatrixPerplexity.item(x,y)).replace('.',',')+"\t")
                    FileDistanceMean.write(str(meanMatrixDistance.item(x,y)).replace('.',',')+"\t")
                    FileDistanceSTD.write(str(sdtMatrixDistance.item(x,y)).replace('.',',')+"\t")
                    FileKLDMean.write(str(meanMatrixKLD.item(x,y)).replace('.',',')+"\t")
                    FileKLDSTD.write(str(sdtMatrixKLD.item(x,y)).replace('.',',')+"\t")
                FilePLLTrainMean.write("\n")
                FilePLLTrainSTD.write("\n")
                FilePLLTestMean.write("\n")
                FilePLLTestSTD.write("\n")
                FilePerplexityMean.write("\n")
                FilePerplexitySTD.write("\n")
                FileDistanceMean.write("\n")
                FileDistanceSTD.write("\n")
                FileKLDMean.write("\n")
                FileKLDSTD.write("\n")
            FilePLLTrainMean.close()
            FilePLLTrainSTD.close()
            FilePLLTestMean.close()
            FilePLLTestSTD.close()
            FilePerplexityMean.close()
            FilePerplexitySTD.close()
            FileDistanceMean.close()
            FileDistanceSTD.close()
            FileKLDMean.close()
            FileKLDSTD.close()

        import sys; sys.exit("Error message")



    if promediar == 1:
        listaPLLTrain = []
        listaPLLTest = []
        listaPerplexity = []
        listaDistance = []
        listaKLD = []
        for exp in expNumber:
            for top in listaTop:
                PhiLLTrainMatrix  = numpy.zeros((5,5))
                PhiLLTestMatrix  = numpy.zeros((5,5))
                PerplexityMatrix  = numpy.zeros((5,5))
                DistanceMatrix  = numpy.zeros((5,5))
                KLDMatrix  = numpy.zeros((5,5))
                expFileTest = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTest_top"+str(top)+".txt","r");
                FilePLLTrain = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTrain_top"+str(top)+".txt","r");
                FilePLLTest = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTest_top"+str(top)+".txt","r");
                FilePerplexity = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Perplexity_top"+str(top)+".txt","r");
                FileDistance = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Distance_top"+str(top)+".txt","r");
                FileKLD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/KLD_top"+str(top)+".txt","r");

                x=0;y=0;
                for line,line2,line3,line4,line5, in zip(FilePLLTrain,FilePLLTest,FilePerplexity,FileDistance,FileKLD):
                    for l1,l2,l3,l4,l5 in zip(line.strip().split("\t"),line2.strip().split("\t"),line3.strip().split("\t"),line4.strip().split("\t"),line5.strip().split("\t")):
                        PhiLLTrainMatrix[x][y] = float(str(l1).replace(",","."));
                        PhiLLTestMatrix[x][y] = float(str(l2).replace(",","."));
                        PerplexityMatrix[x][y] = float(str(l3).replace(",","."));
                        DistanceMatrix[x][y] = float(str(l4).replace(",","."));
                        KLDMatrix[x][y] = float(str(l5).replace(",","."));
                        y= y+1;
                    y=0;
                    x=x+1;
                FilePLLTrain.close();
                FilePLLTest.close();
                FilePerplexity.close();
                FileDistance.close();
                FileKLD.close();

                listaPLLTrain.append(PhiLLTrainMatrix)
                listaPLLTest.append(PhiLLTestMatrix)
                listaPerplexity.append(PerplexityMatrix)
                listaDistance.append(DistanceMatrix)
                listaKLD.append(KLDMatrix)

        meanMatrixPLLTrain  = numpy.zeros((5,5))
        sdtMatrixPLLTrain = numpy.zeros((5,5))
        for x in range(5):
            for y in range(5):
                #print  float(listaPLLTrain[0][x][y]),float(listaPLLTrain[1][x][y]), float(listaPLLTrain[2][x][y]), float(listaPLLTrain[3][x][y]), float(listaPLLTrain[4][x][y])
                if float(listaPLLTrain[0][x][y]) != 0.0:
                    if float(listaPLLTrain[1][x][y]) != 0.0:
                        if float(listaPLLTrain[2][x][y]) != 0.0:
                            if float(listaPLLTrain[3][x][y]) != 0.0:
                                if float(listaPLLTrain[4][x][y]) != 0.0:
                                    meanMatrixPLLTrain[x][y] = numpy.mean([float(listaPLLTrain[0][x][y]),float(listaPLLTrain[1][x][y]),float(listaPLLTrain[2][x][y]),float(listaPLLTrain[3][x][y]),float(listaPLLTrain[4][x][y])])
                                    sdtMatrixPLLTrain[x][y] = numpy.std([float(listaPLLTrain[0][x][y]),float(listaPLLTrain[1][x][y]),float(listaPLLTrain[2][x][y]),float(listaPLLTrain[3][x][y]),float(listaPLLTrain[4][x][y])])

        print meanMatrixPLLTrain
        print sdtMatrixPLLTrain


        meanMatrixPLLTest  = numpy.zeros((5,5))
        sdtMatrixPLLTest = numpy.zeros((5,5))
        for x in range(5):
            for y in range(5):
                #print  float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]), float(listaPLLTest[2][x][y]), float(listaPLLTest[3][x][y]), float(listaPLLTest[4][x][y])
                if float(listaPLLTest[0][x][y]) != 0.0:
                    if float(listaPLLTest[1][x][y]) != 0.0:
                        if float(listaPLLTest[2][x][y]) != 0.0:
                            if float(listaPLLTest[3][x][y]) != 0.0:
                                if float(listaPLLTest[4][x][y]) != 0.0:
                                    meanMatrixPLLTest[x][y] = numpy.mean([float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]),float(listaPLLTest[2][x][y]),float(listaPLLTest[3][x][y]),float(listaPLLTest[4][x][y])])
                                    sdtMatrixPLLTest[x][y] = numpy.std([float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]),float(listaPLLTest[2][x][y]),float(listaPLLTest[3][x][y]),float(listaPLLTest[4][x][y])])

        print meanMatrixPLLTest
        print sdtMatrixPLLTest

        meanMatrixPerplexity  = numpy.zeros((5,5))
        sdtMatrixPerplexity = numpy.zeros((5,5))
        for x in range(5):
            for y in range(5):
                #print  float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]), float(listaPLLTest[2][x][y]), float(listaPLLTest[3][x][y]), float(listaPLLTest[4][x][y])
                if float(listaPerplexity[0][x][y]) != 0.0:
                    if float(listaPerplexity[1][x][y]) != 0.0:
                        if float(listaPerplexity[2][x][y]) != 0.0:
                            if float(listaPerplexity[3][x][y]) != 0.0:
                                if float(listaPerplexity[4][x][y]) != 0.0:
                                    meanMatrixPerplexity[x][y] = numpy.mean([float(listaPerplexity[0][x][y]),float(listaPerplexity[1][x][y]),float(listaPerplexity[2][x][y]),float(listaPerplexity[3][x][y]),float(listaPerplexity[4][x][y])])
                                    sdtMatrixPerplexity[x][y] = numpy.std([float(listaPerplexity[0][x][y]),float(listaPerplexity[1][x][y]),float(listaPerplexity[2][x][y]),float(listaPerplexity[3][x][y]),float(listaPerplexity[4][x][y])])

        print meanMatrixPerplexity
        print sdtMatrixPerplexity


        meanMatrixDistance  = numpy.zeros((5,5))
        sdtMatrixDistance = numpy.zeros((5,5))
        for x in range(5):
            for y in range(5):
                #print  float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]), float(listaPLLTest[2][x][y]), float(listaPLLTest[3][x][y]), float(listaPLLTest[4][x][y])
                if float(listaDistance[0][x][y]) != 0.0:
                    if float(listaDistance[1][x][y]) != 0.0:
                        if float(listaDistance[2][x][y]) != 0.0:
                            if float(listaDistance[3][x][y]) != 0.0:
                                if float(listaDistance[4][x][y]) != 0.0:
                                    meanMatrixDistance[x][y] = numpy.mean([float(listaDistance[0][x][y]),float(listaDistance[1][x][y]),float(listaDistance[2][x][y]),float(listaDistance[3][x][y]),float(listaDistance[4][x][y])])
                                    sdtMatrixDistance[x][y] = numpy.std([float(listaDistance[0][x][y]),float(listaDistance[1][x][y]),float(listaDistance[2][x][y]),float(listaDistance[3][x][y]),float(listaDistance[4][x][y])])

        print meanMatrixDistance
        print sdtMatrixDistance

        meanMatrixKLD  = numpy.zeros((5,5))
        sdtMatrixKLD = numpy.zeros((5,5))
        for x in range(5):
            for y in range(5):
                #print  float(listaPLLTest[0][x][y]),float(listaPLLTest[1][x][y]), float(listaPLLTest[2][x][y]), float(listaPLLTest[3][x][y]), float(listaPLLTest[4][x][y])
                if float(listaKLD[0][x][y]) != 0.0:
                    if float(listaKLD[1][x][y]) != 0.0:
                        if float(listaKLD[2][x][y]) != 0.0:
                            if float(listaKLD[3][x][y]) != 0.0:
                                if float(listaKLD[4][x][y]) != 0.0:
                                    meanMatrixKLD[x][y] = numpy.mean([float(listaKLD[0][x][y]),float(listaKLD[1][x][y]),float(listaKLD[2][x][y]),float(listaKLD[3][x][y]),float(listaKLD[4][x][y])])
                                    sdtMatrixKLD[x][y] = numpy.std([float(listaKLD[0][x][y]),float(listaKLD[1][x][y]),float(listaKLD[2][x][y]),float(listaKLD[3][x][y]),float(listaKLD[4][x][y])])

        print meanMatrixKLD
        print sdtMatrixKLD

        FilePLLTrainMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PHLLTrainMean.txt","w");
        FilePLLTrainSTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PHLLTrainSTD.txt","w");
        FilePLLTestMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PHLLTestMean.txt","w");
        FilePLLTestSTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PHLLTestSTD.txt","w");
        FilePerplexityMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PerplexityMean.txt","w");
        FilePerplexitySTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/PerplexitySTD.txt","w");
        FileDistanceMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/DistanceMean.txt","w");
        FileDistanceSTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/DistanceSTD.txt","w");
        FileKLDMean = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/KLDMean.txt","w");
        FileKLDSTD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/KLDSTD.txt","w");


        for x in range (5):
            for y in range(5):
                FilePLLTrainMean.write(str(meanMatrixPLLTrain.item(x,y)).replace('.',',')+"\t")
                FilePLLTrainSTD.write(str(sdtMatrixPLLTrain.item(x,y)).replace('.',',')+"\t")
                FilePLLTestMean.write(str(meanMatrixPLLTest.item(x,y)).replace('.',',')+"\t")
                FilePLLTestSTD.write(str(sdtMatrixPLLTest.item(x,y)).replace('.',',')+"\t")
                FilePerplexityMean.write(str(meanMatrixPerplexity.item(x,y)).replace('.',',')+"\t")
                FilePerplexitySTD.write(str(sdtMatrixPerplexity.item(x,y)).replace('.',',')+"\t")
                FileDistanceMean.write(str(meanMatrixDistance.item(x,y)).replace('.',',')+"\t")
                FileDistanceSTD.write(str(sdtMatrixDistance.item(x,y)).replace('.',',')+"\t")
                FileKLDMean.write(str(meanMatrixKLD.item(x,y)).replace('.',',')+"\t")
                FileKLDSTD.write(str(sdtMatrixKLD.item(x,y)).replace('.',',')+"\t")
            FilePLLTrainMean.write("\n")
            FilePLLTrainSTD.write("\n")
            FilePLLTestMean.write("\n")
            FilePLLTestSTD.write("\n")
            FilePerplexityMean.write("\n")
            FilePerplexitySTD.write("\n")
            FileDistanceMean.write("\n")
            FileDistanceSTD.write("\n")
            FileKLDMean.write("\n")
            FileKLDSTD.write("\n")
        FilePLLTrainMean.close()
        FilePLLTrainSTD.close()
        FilePLLTestMean.close()
        FilePLLTestSTD.close()
        FilePerplexityMean.close()
        FilePerplexitySTD.close()
        FileDistanceMean.close()
        FileDistanceSTD.close()
        FileKLDMean.close()
        FileKLDSTD.close()

        import sys; sys.exit("Error message")

    for exp in expNumber:
        files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/*")
        for f in files:
            #print f
            os.remove(f)

    for exp in expNumber:
        expFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Results.txt","w");

        expFile.write("topic"+"\t"+" words"+"\t"+" alpha"+"\t"+" beta"+"\t"+"trainP"+"\t"+"testP"+"\t"+"dist"+"\t"+"kld"+"\n")
        for top in listaTop:

            FileParameters = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/Parameters_top"+str(top)+".txt","w");
            FileParameters.write(str(top)+"\n");
            FileParameters.write(str(numWords)+"\n");
            FileParameters.write(str(TrainDocs)+"\n")




            FilePLLTrain = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTrain_top"+str(top)+".txt","w");
            FilePLLTrain = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTrain_top"+str(top)+".txt","ab");

            FilePLLTest = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTest_top"+str(top)+".txt","w");
            FilePLLTest = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/PHLLTest_top"+str(top)+".txt","ab");

            FilePerplexity = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Perplexity_top"+str(top)+".txt","w");
            FilePerplexity = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Perplexity_top"+str(top)+".txt","ab");

            FileDistance = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Distance_top"+str(top)+".txt","w");
            FileDistance = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/Distance_top"+str(top)+".txt","ab");

            FileKLD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/KLD_top"+str(top)+".txt","w");
            FileKLD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/RESULTS"+str(exp)+"/KLD_top"+str(top)+".txt","ab");

            alpha = float(50/float(top));
            listaAlpha = [float(alpha/100),float(alpha/10),alpha, float(alpha*10), float(alpha*100)]
            FileParameters.write(" ".join(map(str,listaAlpha))+"\n")
            beta = float(200/float(numWords));
            listaBeta = [float(beta/100),float(beta/10), beta, float(beta*10), float(beta*100)]
            FileParameters.write(" ".join(map(str,listaBeta))+"\n")
            FileParameters.close()
            for alph in listaAlpha:

                for bet in listaBeta:
                    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/LDA/ModelMatrix/*")
                    for f in files:
                        os.remove(f)
                    print "Entra a LDA con LDA(",top, numWords, alph, bet,")"
                    [trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld]=LDA(top, numWords, alph, bet);

                    expFile.write(str(str(top)+"\t"+str(numWords)+"\t"+str(alph)+"\t"+str(bet)+"\t").replace('.',','))
                    expFile.write(str(str(trainPerp)+"\t"+str(float(trainPerp)/float(top))+"\t"+str(testPerp)+"\t"+str(float(testPerp)/float(top))+"\t"+str(Perplexity_Emp_Final)+"\t"+str(Distance_Emp_Final)+"\t"+str(kld)+"\n").replace('.',','))

                    FilePLLTrain.write(str(str(float(trainPerp)/float(top))+"\t").replace('.',','))
                    FilePLLTest.write(str(str(float(testPerp)/float(top))+"\t").replace('.',','))
                    FilePerplexity.write(str(str(Perplexity_Emp_Final)+"\t").replace('.',','))
                    FileDistance.write(str(str(Distance_Emp_Final)+"\t").replace('.',','))
                    FileKLD.write(str(str(kld)+"\t").replace('.',','))
                FilePLLTrain.write(str("\n"))
                FilePLLTest.write(str("\n"))
                FilePerplexity.write(str("\n"))
                FileDistance.write(str("\n"))
                FileKLD.write(str("\n"))
            FilePLLTrain.close()
            FilePLLTest.close()
            FilePerplexity.close()
            FileDistance.close()
            FileKLD.close()
        expFile.close()







if method == "ADABOOST_V2":
    print "ENTRA FUNC ADABOOST"
    ADABOOST_V2(numTopics, numWords, float(alpha/100),float(beta))

if method == "ADABOOST":
    print "ENTRA FUNC ADABOOST"
    ADABOOST(numTopics, numWords, alpha, beta)

if method == "ADABOOSTRep":
    import os
    import glob
    #expNumber = [1,2,3,4,5]
    expNumber = [1]
    itNumber = [3];
    if dataset == "Dataset4": listaTop = [2];
    if dataset == "Dataset5": listaTop = [40];#30
    if dataset == "Dataset6": listaTop = [400];#450
    #listaAlpha = [float(alpha/10), alpha, float(alpha*10)]
    #listaBeta = [float(beta/10), beta, float(beta*10)]
    listaBeta = [beta]

    print "ENTRA FUNC ADABOOST REPEATED"
    #ADABOOST(numTopics, numWords, alpha, beta, 2)

    for exp in expNumber:
        files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/ADABOOST/RESULTS"+str(exp)+"/*")
        for f in files:
            os.remove(f)

    for exp in expNumber:
        print "****Experimento Numero: ", exp, " ****"
        expFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/ADABOOST/RESULTS"+str(exp)+"/Results.txt","w");
        for top in listaTop:
            print top
            alpha = float(50/float(top));
            listaAlpha = [float(alpha/100)]
            for alph in listaAlpha:
                print alph
                beta = float(200/float(numWords));
                listaBeta = [beta]
                for bet in listaBeta:
                    print bet
                    print top, numWords, alph, bet

                    for it in itNumber:
                        files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/ADABOOST/ModelMatrix/*")
                        for f in files:
                            os.remove(f)
                        [trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld] = ADABOOST(top, numWords, alph, bet, it)
                        print top, numWords, alph, bet
                        print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld

                        expFile.write("num iter: "+str(it)+"\n")
                        expFile.write("topics: "+str(top)+" words: "+str(numWords)+" alpha: "+str(alph)+" beta: "+str(bet)+"\n")
                        expFile.write("trainPerp: "+str(trainPerp)+"\n")
                        expFile.write("testPerp: "+str(testPerp)+"\n")
                        expFile.write("Perp_Emp_Final: "+str(Perplexity_Emp_Final)+"\n")
                        expFile.write("Dist_Emp_Final: "+str(Distance_Emp_Final)+"\n")
                        expFile.write("kld: "+str(kld)+"\n")
                        expFile.write("\n")
                        expFile.write("\n")

                        print "=======DDD======"
                        print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld
                        print "=======DDD======"

        expFile.close()




if method == "BAGGING":
    print "ENTRA FUNC BAGGING"
    BAGGING(numTopics, numWords, alpha, beta)

if method == "BAGGINGRep":
    import os
    import glob
    #expNumber = [1,2,3,4,5]
    expNumber = [1]
    itNumber = [2,4,6,8,10];
    if dataset == "Dataset4": listaTop = [2,2];
    if dataset == "Dataset5": listaTop = [5,10,15];#30
    if dataset == "Dataset6": listaTop = [50,100,150];#300
    if dataset == "Dataset7": listaTop = [200,300,400];#450
    #listaAlpha = [float(alpha/10), alpha, float(alpha*10)]
    #listaBeta = [float(beta/10), beta, float(beta*10)]
    listaBeta = [beta]

    print "ENTRA FUNC BAGGING == "

    for exp in expNumber:
        files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/BAGGING/RESULTS"+str(exp)+"/*")
        for f in files:
            #print f
            os.remove(f)

    for exp in expNumber:
        print "exp"
        expFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/BAGGING/RESULTS"+str(exp)+"/Results.txt","w");
        for top in listaTop:
            print top
            alpha = float(50/float(top));
            listaAlpha = [float(alpha/10), alpha, float(alpha*10)]
            for alph in listaAlpha:
                print alph
                for bet in listaBeta:
                    print bet
                    print top, numWords, alph, bet

                    for it in itNumber:
                        files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/BAGGING/ModelMatrix/*")
                        for f in files:
                            os.remove(f)
                        [trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld] = BAGGING(top, numWords, alph, bet, it)
                        print top, numWords, alph, bet
                        print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld

                        expFile.write("num iter: "+str(it)+"\n")
                        expFile.write("topics: "+str(top)+" words: "+str(numWords)+" alpha: "+str(alph)+" beta: "+str(bet)+"\n")
                        expFile.write("trainPerp: "+str(trainPerp)+"\n")
                        expFile.write("testPerp: "+str(testPerp)+"\n")
                        expFile.write("Perp_Emp_Final: "+str(Perplexity_Emp_Final)+"\n")
                        expFile.write("Dist_Emp_Final: "+str(Distance_Emp_Final)+"\n")
                        expFile.write("kld: "+str(kld)+"\n")
                        expFile.write("\n")
                        expFile.write("\n")

                        print "=======DDD======"
                        print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld
                        print "=======DDD======"

        expFile.close()












#   from pylab import *
#   import matplotlib.pyplot as plt
#   import matplotlib.image as mpimg
#   import numpy as np
#   img = matrix_Z_Words_Total
#   #print img
#   img = 1- matrix_Z_Words_Total
#   #print img
#   imgplot = plt.imshow(img, interpolation='nearest', vmin = 0, vmax = 1, cmap=mpl.cm.gray)
#   plt.grid(True)
#   plt.show()



if method == "2ND_ADABOOST":

        import time
        import glob
        import numpy

        f = open("C:/Mallet/testCMD/"+file,"r")
        docs = 0;
        for line in f: docs = docs + 1;
        f.close();
        prob = float(1/float(docs))
        setProbabilities = [];
        for doc in range(docs): setProbabilities.append(prob);
        [sampleFile,number, setProbabilities] = generateSampleFileM(file, setProbabilities)
        #print file; print sampleFile;#print folder
        folder = file.split("\\")[0];

        #=====SE CREA ARCHIVO (words_count_file_gral.txt) CCON TODAS LAS PALABRAS DEL CORPUS====
        formatingFiles(numTopics,file,folder,alpha, beta)
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

        priorsDocsDGral = priorsDoc;
        priorsCorpusGral = priorsCorpus;
        matrixDocTermsGral = matrixDocTerms; matrixDocTermFinalGral = matrixDocTermFinal; matrixProbTermsDocGral = matrixProbTermsDoc;

        wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
        wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file_gral.txt","w");

        numTerms = 0;
        for elem in wordsFileA:
            wordsFileB.write(elem);
            numTerms = numTerms + 1;
        wordsFileA.close();
        wordsFileB.close();



        machinesPerplexity = [];
        confidenceParametersAlphas = [];
        confidenceAlpha = 0;

        #print setProbabilities
        #import sys; sys.exit("Error message")
        numIterations = 3;
        cont = 1;
        topicsTotal = 0;
        while cont <= numIterations:


            print "Iteracion "+ str(cont);
            if cont > 1:
                [sampleFile,number, setProbabilities] = generateSampleFileM(file, vectorNewProbabilities)
            successful = False;
            rounds = 0;
            time.sleep(0.2);random.seed(time.time());numTopics = random.choice(numberOfTopics); alpha = random.choice(alphaChoice); rounds =0;
            while not successful:
                #print "Numero Topicos, Rounds"
                #print numTopics, alpha, beta, rounds
                if rounds % 5 == 0: print "mas de 2"; numTopics = int(numTopics - numTopics*0.2)+ 1; print numTopics;
                if rounds == 10:  time.sleep(0.2);random.seed(time.time());numTopics = random.choice(numberOfTopics); alpha = random.choice(alphaChoice); rounds =0;

                #numTopics = 3;
                print "Numero Topicos, Rounds"
                print numTopics, rounds;

                print "Entrenando con Muestra";
                formatingFiles(numTopics,sampleFile,folder,alpha, beta)
                [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,sampleFile,folder)
                [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)

                Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)

                print Distance_Emp
                if Distance_Emp < 0.5:
                    print Distance_Emp
                    confidenceAlpha  = float(0.5*math.log((1-Distance_Emp)/Distance_Emp));
                    successful = True;
                    confidenceParametersAlphas.append(confidenceAlpha);
                #import sys; sys.exit("Error message")

                #if float(Perplexity_Prob*2)> float(Perplexity_Emp):
                #    print Perplexity_Prob, Perplexity_Emp
                #    ffff =  math.fabs(float(Perplexity_Prob-Perplexity_Emp));
                #    if ffff == 0: continue;
                #    if float(ffff)< float(Perplexity_Prob/2):
                #        confidenceAlpha  = float(0.5*math.log((Perplexity_Prob-ffff)/ffff));
                #        successful = True;
                #        confidenceParametersAlphas.append(confidenceAlpha);


                rounds = rounds + 1;

            print "Probando con conjunto de Training"
            #print numTopics, alpha, beta
            formatingFiles(numTopics,file,folder,alpha, beta)
            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)
            [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
            print "Perp Maquina"
            machinesPerplexity.append(Perplexity_Emp);

            vectorDifEmpProb  =[];
            vectorNewProbabilities = [];

            for l1, l2 in zip(probVectorEmp,probVectorProb): vectorDifEmpProb.append(l1-l2);
            for l1, l2 in zip(vectorDifEmpProb,setProbabilities): vectorNewProbabilities.append(float(l2)*math.exp(float(l1)*confidenceAlpha))

            sumProb = 0;

            for elem in vectorNewProbabilities: sumProb = sumProb + float(elem)

            vectorNewProbabilities[:] = [x/sumProb for x in vectorNewProbabilities]

            print vectorNewProbabilities;
            print "TOP WORD AND PRIORS"
            print matrixTopWord.transpose();
            rows = matrixTopWord.transpose().shape[0]; cols = matrixTopWord.transpose().shape[1]
            print numpy.array([matrixTopWord.sum(axis=1)])/rows
            print "TOP WORD AND PRIORS"
            testArr = numpy.array([matrixTopWord.sum(axis=1)])/rows;
            flag = 0;
            for elem in testArr[0]:
                if float(elem) == 0:
                    flag = 1;
                    continue;

            #import sys; sys.exit("Error message")
            if flag == 0:
                cont = cont + 1;
                TOTAL_DTMEmp.append(matrixDocTermFinal)
                TOTAL_TOP_WORD_Emp.append(matrixTopWord.transpose());
                TOTAL_TOP_PRIORS.append(numpy.array([matrixTopWord.sum(axis=1)])/rows);
                TOTAL_DOC_TOP_Emp.append(matrixDocTop);




        sumTopPriors = 0;
        TOTAL_TOP_PRIORS_NORM = [];

        for elem in TOTAL_TOP_PRIORS:
            for num in elem[0]:
                sumTopPriors = sumTopPriors+ float(num);

        for elem in TOTAL_TOP_PRIORS:
            for num in elem[0]:
                TOTAL_TOP_PRIORS_NORM.append(float(num)/float(sumTopPriors))

        print TOTAL_TOP_PRIORS_NORM




        #print TOTAL_TOP_WORD_Emp;
        #print TOTAL_TOP_PRIORS
        #print confidenceParametersAlphas
        #print TOTAL_DTMEmp;
        #print topicsTotal;

        topicsTotal = len(TOTAL_TOP_PRIORS_NORM);
        print "totaltOPIC" + str(topicsTotal)

        rows = matrixDocTerms.shape[0]

        matrixWords_Z_Total = numpy.zeros((rows,topicsTotal))
        #print matrixWords_Z_Total


        total_cols = 0;
        for l1,l2,l3 in zip(TOTAL_TOP_WORD_Emp,TOTAL_TOP_PRIORS,confidenceParametersAlphas):
            rows = l1.shape[0]; cols = l1.shape[1]
            total_rows = 0;
            for i in range(rows):
                for j in range(cols):
                    matrixWords_Z_Total[total_rows][total_cols+j] = float(l3)*float(l1[i][j]);
                total_rows = total_rows + 1;
            total_cols = total_cols + cols;
        print matrixWords_Z_Total



        rows = matrixDocTop.shape[0]
        matrixDoc_Z_Total = numpy.zeros((rows,topicsTotal))
        total_cols = 0;
        for elem in TOTAL_DOC_TOP_Emp:
            rows = elem.shape[0]; cols = elem.shape[1]
            total_rows = 0;
            for i in range(rows):
                for j in range(cols):
                    matrixDoc_Z_Total[total_rows][total_cols+j] = float(elem[i][j]);
                total_rows = total_rows + 1;
            total_cols = total_cols + cols;

        #print matrixDoc_Z_Total
        matrixDoc_Z_Total = matrixDoc_Z_Total/numpy.array([matrixDoc_Z_Total.sum(axis=1)]).transpose()
        print matrixDoc_Z_Total



        matAux = numpy.zeros((len(TOTAL_TOP_PRIORS_NORM),len(priorsCorpusGral)));

        x = 0; y = 0;
        print TOTAL_TOP_PRIORS_NORM
        print priorsCorpusGral
        for x1 in TOTAL_TOP_PRIORS_NORM:#P(META)
            for y1 in priorsCorpusGral:#P(DOC)M
                matAux[x][y] =  float(y1)/float(x1)
                y = y + 1;
            x = x + 1;
            y = 0;

        #print matAux;

        matrixWords_Z_Total = matrixWords_Z_Total/numpy.array([matrixWords_Z_Total.sum(axis=1)]).transpose()
        matrix_Z_Words_Total =  matrixWords_Z_Total.transpose()*matAux
        matrix_Z_Words_Total = matrix_Z_Words_Total/numpy.array([matrix_Z_Words_Total.sum(axis=1)]).transpose()
        #print matrixWords_Z_Total
        #print matrix_Z_Words_Total
        #import sys; sys.exit("Error message")

        wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
        i = 0;
        #=====SE GUARDA LA MATRIZ DE PROBABILIDADES CONDICIONALES=====
        wordList  =[];
        allWordsString = "" #=====STRING DONDE SE CONCATENARA CADA FILA
        for elem in wordsFileE:
            wordList.append(str(elem).split(" ")[1])
            allWordsString = allWordsString+" "+str(elem).split(" ")[1];
        wordsFileE.close();

        rows = matrix_Z_Words_Total.shape[0]; cols = matrix_Z_Words_Total.shape[1]
        wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/matrixPeudoDocumentWords_Adaboost.txt","w");


        #print matrixWords_Z_Total.transpose()
        NUMERODOCUMENTOS = 1;
        for i in range(rows):
            wordsFileF.write("D T:")
            NUMEROPALABRAS = 0;
            for j in range(cols):
                if j == 0:
                    wordsFileF.write(allWordsString)

                cardinality = int(matrix_Z_Words_Total.item(i,j)*10);
                for word in range(cardinality):
                    wordsFileF.write(" "+str(wordList[j]));
                    NUMEROPALABRAS = NUMEROPALABRAS + 1;

            #print NUMERODOCUMENTOS, NUMEROPALABRAS
            if NUMEROPALABRAS == 0:
                wordsFileF.write(allWordsString+"\n")
            else:
                wordsFileF.write("\n")
            NUMERODOCUMENTOS = NUMERODOCUMENTOS + 1;

        wordsFileF.close();


        #===== NOTA (MATRIZ Z W)
        #=====Z, documents
        #=====W, words
        #=====Y, topics
        new_file = file.split("\\")[0]+"/matrixPeudoDocumentWords_Adaboost.txt";
        print "===Inicio LDA para pseudo documento==="
        numTopics  = random.choice(numberOfTopics); num_top_words = 20; alpha = random.choice(alphaChoice); beta = random.choice(betaChoice)
        print numTopics, alpha, beta
        formatingFiles(numTopics,new_file,folder,alpha, beta)
        [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,new_file,folder)

        # matrixDocTop P(Y/Z)
        #print matrixDocTop
        #print priorsCorpus, priorsDoc

        zTopicsPrior = TOTAL_TOP_PRIORS_NORM
        yTopicsPrior = numpy.array([matrixDocTop.sum(axis=0)])/matrixDocTop.shape[0]

        #print zTopicsPrior, yTopicsPrior
        #import sys; sys.exit("Error message")


        matAux = numpy.zeros((len(yTopicsPrior[0]),len(zTopicsPrior)))

        x = 0; y = 0;
        for elemA in yTopicsPrior[0]:
            for elemB in zTopicsPrior:
                matAux[x][y] = elemB/elemA;
                y = y + 1;
            x = x + 1;
            y = 0;

        #P(Z/Y) = P(Y/Z) Transpose
        matrixTopDoc = matrixDocTop.transpose()*matAux

        #print matrixTopDoc


        #===P(Y/d)
        #=== matrixDoc_Z_Total P(Z/D)
        #=== matrizDocTop P(Y/Z)
        matrixDocTopY = numpy.dot(matrixDoc_Z_Total,matrixDocTop);

        print matrixDoc_Z_Total
        print matrixDocTop


        #=== P(W/Y)

        matAux = numpy.zeros((len(yTopicsPrior[0]),len(priorsCorpus)))

        print "====="
        print "====="
        print priorsCorpus
        print yTopicsPrior
        x = 0; y = 0;
        for elemA in yTopicsPrior[0]:
            for elemB in priorsCorpus:
                print elemA, elemB
                matAux[x][y] = elemB/elemA;
                y = y + 1;
            x = x + 1;
            y = 0;
        #print matAux
        matrixWordTop = matrixTopWord.transpose();
        matrixTopWord =  matrixTopWord*matAux




        print matrixWordTop
        print matrixTopWord


        #===P(W/D)
        matrixDocWords = numpy.dot(matrixDocTopY,matrixTopWord);
        print "#===P(W/D)"
        print matrixDocWords
        #print matrixDocTopY
        #print matrixTopWord



        matAux = numpy.zeros((len(priorsCorpusGral),len(priorsDocsDGral)))

        x = 0; y = 0;
        for elemA in priorsCorpusGral:
            for elemB in priorsDocsDGral:
                #print elemA, elemB
                matAux[x][y] = elemB/elemA;
                y = y + 1;
            x = x + 1;
            y = 0;

        print "#===P(D/W)"
        print matAux

        matrixWordDocs = matrixDocWords.transpose()*matAux

        print matrixDocTerms; print matrixProbTermsDoc;

        matrixWordDocs = matrixWordDocs/numpy.array([matrixWordDocs.sum(axis=1)]).transpose()
        print matrixWordDocs
        print matrixDocTermsGral

        from pylab import *
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np
        img = matrix_Z_Words_Total
        print img
        img = 1- matrix_Z_Words_Total
        print img
        imgplot = plt.imshow(img, interpolation='nearest', vmin = 0, vmax = 1, cmap=mpl.cm.gray)
        plt.grid(True)
        plt.show()


        import sys; sys.exit("Error message")







        suma = 0;
        for l1, l2 in zip(confidenceParametersAlphas,TOTAL_DTMEmp):
            suma = suma + l1*l2
        print suma

        suma =suma/numpy.array([suma.sum(axis=1)]).transpose()
        print suma
        [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,suma,matrixProbTermsDoc)
        print machinesPerplexity
        print "Perp Prob"
        print Perplexity_Prob
        print "Perp Emp"
        print Perplexity_Emp

        from pylab import *
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np
        img = suma
        print img
        img = 1- suma
        print img
        imgplot = plt.imshow(img, interpolation='nearest', vmin = 0, vmax = 1, cmap=mpl.cm.gray)
        plt.grid(True)
        plt.show()






if method == "TOPIC_MODELS_ENSEMBLE":

    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
    import glob
    import numpy
    folder = file.split("\\")[0];

    #=====SE DETERMINA EL NUMERO DE SUBCONJUNTOS Y LA CANTIDAD DE DOCUMENTOS POR SUBCONJUNTO====
    subsetNum = 2;
    ssElemNum = cont//subsetNum;
    numDocs = cont;

    #=====SE ELIMINAN LOS ARCHIVOS CREADOS ANTERIORMENTE====
    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/SubSetFile*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/matrixTopWord_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/subSet_Words_Matrix_*.txt")
    for subfile in files: os.remove(subfile)

    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/matrixDocTop_*.txt")
    for subfile in files: os.remove(subfile)


    #=====SE DIVIDE EL CORPUS Y SE CREAN LOS ARCHIVOS SUBCONJUNTO====
    i = 0;
    j = 0;
    f = open("C:/Mallet/testCMD/"+file,"r");
    elemWritten = 0;

    subsetFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/SubSetFile"+(letters[i])+".txt","w");
    for elem in f:
        if elemWritten <= cont:
            if j < ssElemNum: #=====SE ESCRIBE MIENTRAS NO SE SUPERE EL LIMITE DE ELEMENTOS====
                elemWritten = elemWritten+ 1;
                subsetFile.write(elem)
                j = j + 1;
            else:             #=====UNA VEZ SUPERADO EL LIMITE SE CIERRA EL ARCHIVO Y SE CREA UNO NUEVO====
                subsetFile.close();
                i = i + 1;
                j = 0
                subsetFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/SubSetFile"+(letters[i])+".txt","w");
                elemWritten = elemWritten+ 1;
                subsetFile.write(elem.replace("D T:","D1 T:"))
                j = j + 1;

    subsetFile.close();



    #=====SE CREA ARCHIVO (words_count_file_gral.txt) CCON TODAS LAS PALABRAS DEL CORPUS====
    formatingFiles(numTopics,file,folder,alpha, beta)
    [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,file,folder)

    priorsDocsDGral = priorsDoc
    matrixDocTermsGral = matrixDocTerms; matrixDocTermFinalGral = matrixDocTermFinal; matrixProbTermsDocGral = matrixProbTermsDoc;

    priorsCorpusGral = priorsCorpus;

    wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
    wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file_gral.txt","w");

    numTerms = 0;
    for elem in wordsFileA:
        wordsFileB.write(elem);
        numTerms = numTerms + 1;
    wordsFileA.close();
    wordsFileB.close();


    k = 0;
    #=====SE REALIZA LDA PARA CADA UNO DE LOS SUBCONJUNTOS EN LOS ARCHIVOS (SubSetFile*.txt)====
    files = glob.glob("C:/Mallet/testCMD/"+file.split("\\")[0]+"/SubSetFile*.txt")
    topics = 0;
    topicsPerSubset = []
    for subfile in files:
        subfile = folder+"\\"+subfile.split("\\")[1]
        flag = 0;
        rounds = 0;
        while flag == 0:
            numTopics  = random.choice(numberOfTopics); num_top_words = 20; alpha = random.choice(alphaChoice); beta = random.choice(betaChoice)
            print "Numero Topicos, Rounds"
            print numTopics, alpha, beta, rounds
            #numTopics = int(float(1/float(subsetNum))*numTopics)
            print numTopics, alpha, beta, rounds

            #if rounds % 5 == 0: print "mas de 2"; numTopics = int(numTopics - numTopics*0.2)+ 1; print numTopics;
            #if rounds == 10:  time.sleep(0.2);random.seed(time.time());numTopics = random.choice(numberOfTopics); alpha = random.choice(alphaChoice); rounds =0;



            formatingFiles(numTopics,subfile,folder,alpha, beta)
            wordsFileA = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
            wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file_"+(letters[k])+".txt","w");
            for elem in wordsFileA: #=====SE GUARDA CADA SALIDA DEL CONTEO DE PALABRAS EN ARCHIVOS DIFERENTES (words_count_file_*.txt)=====
                wordsFileB.write(elem);
            wordsFileA.close();
            wordsFileB.close();

            flag = 1;

            [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,subfile,folder)

            sumsArr = numpy.array([matrixTopWord.sum(axis=1)])
            print sumsArr[0]
            for elem in sumsArr[0]:
                if float(elem) == 0.0:
                    print letters[k];
                    flag = 0;
                    break;

            rounds = rounds + 1;

        #print matrixTopWord
        #print sumsArr.transpose();

        rows = matrixTopWord.transpose().shape[0]; cols = matrixTopWord.transpose().shape[1]
        topics = topics + matrixTopWord.transpose().shape[1]
        topicsPerSubset.append(matrixTopWord.transpose().shape[1])
        wordsFileC = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/matrixTopWord_"+(letters[k])+".txt","w");
        for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
            for col in range(cols):
                wordsFileC.write(str(matrixTopWord.transpose().item(row,col))+"\t")
            wordsFileC.write("\n");

        #print matrixDocTop
        rows = matrixDocTop.shape[0]; cols = matrixDocTop.shape[1]
        wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/matrixDocTop_"+(letters[k])+".txt","w");
        for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
            for col in range(cols):
                wordsFileD.write(str(matrixDocTop.item(row,col))+"\t")
            wordsFileD.write("\n");

        wordsFileD.close();
        TOTAL_DTMEmp.append(matrixDocTermFinal)
        TOTAL_TOP_WORD_Emp.append(matrixTopWord.transpose());
        rows = matrixTopWord.transpose().shape[0]; cols = matrixTopWord.transpose().shape[1]
        TOTAL_TOP_PRIORS.append(numpy.array([matrixTopWord.sum(axis=1)])/rows);
        TOTAL_DOC_TOP_Emp.append(matrixDocTop);
        k = k + 1;


    print "SIN PROBLEMAS"


    sumTopPriors = 0;
    TOTAL_TOP_PRIORS_NORM = [];

    for elem in TOTAL_TOP_PRIORS:
        for num in elem[0]:
            sumTopPriors = sumTopPriors+ float(num);

    for elem in TOTAL_TOP_PRIORS:
        for num in elem[0]:
            TOTAL_TOP_PRIORS_NORM.append(float(num)/float(sumTopPriors))

    #print TOTAL_TOP_PRIORS_NORM

    #=====SE GUARDA CREA UN ARCHIVO PARA CADA SUBCONJUNTO DONDE SE GUARDAN=====
    #=====LAS MATRICES Y CADA PALABRA CORRESPONDIENTE (subSet_Words_Matrix_*.txt)=====
    aggrMatrixTermTop = numpy.zeros((numTerms,topics))
    #print aggrMatrixTermTop

    for let in range(k):
        wordsFileB = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file_"+(letters[let])+".txt","r");
        wordsFileC = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/matrixTopWord_"+(letters[let])+".txt","r");
        wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/subSet_Words_Matrix_"+(letters[let])+".txt","w");
        for l1, l2 in zip(wordsFileB, wordsFileC):
            wordsFileD.write(str(l1).split(" ")[1]+"\t"+l2)
            #print l1, l2

        wordsFileB.close();
        wordsFileC.close();
        wordsFileD.close();




    #=====SE CREA LA MATRIZ "FUSIONADA" DE TOPICO PALABRA A PARTIR DE LOS SUBCONJUNTOS
    wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file_gral.txt","r");
    wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/Words_Matrix_Gral.txt","w");
    savedString = ""

    allWordsString = "";

    for elemE in wordsFileE:
        savedString = "" #=====STRING DONDE SE CONCATENARA CADA FILA
        savedString = savedString + str(elemE).split(" ")[1]
        allWordsString = allWordsString+" "+str(elemE).split(" ")[1];

        for let, top in zip(range(k),topicsPerSubset):

            wordsFileG = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/subSet_Words_Matrix_"+(letters[let])+".txt","r");
            flag = 0;
            for elemG in wordsFileG:

                auxList = list(str(elemG).split("\t"));
                print auxList
                if str(elemE).split(" ")[1] == str(auxList[0]): #=====SI LA PALABRA COINCIDE CON LA DEL SUBCONJUNTO
                    del auxList[-1]
                    del auxList[0]
                    savedString = savedString +" "+ ' '.join(auxList) #=====SE AGREGA LA FILA DE LA MATRIZ AL STRING
                    flag = 1;
                    break;
            if flag == 0:
                for t in range(top):
                    savedString = savedString +" "+str(0.0);

            wordsFileG.close();
        wordsFileF.write(savedString+"\n")
    wordsFileE.close();
    wordsFileF.close();

    print allWordsString

    #import sys;sys.exit("Error message")

    #===== SE CREA LA MATRIZ FUSIONADA DE DOCUMENTO ToPICOS==
    aggrMatrixDocsTop = numpy.zeros((numDocs,topics))


    doc = 0;
    top = 0;
    for let in range(k):
        wordsFileD = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/matrixDocTop_"+(letters[let])+".txt","r");
        for elem in wordsFileD:
            lst = str(elem).split("\t")
            del lst[-1]
            for count in range(len(lst)):
                #print doc, top+count, count
                aggrMatrixDocsTop[doc][top+count] = lst[count];
            doc = doc + 1;
        top = top + len(lst)
        wordsFileD.close();

    #print aggrMatrixDocsTop;


    wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/Document_Topic_Gral.txt","w");

    rows = aggrMatrixDocsTop.shape[0]; cols = aggrMatrixDocsTop.shape[1]
    for row in range(rows): #=====SE GUARDA CADA TOPICO PALABRA EN ARCHIVOS DIFERENTES (matrixTopWord_*.txt)=====
        for col in range(cols):
            wordsFileF.write(str(aggrMatrixDocsTop.item(row,col))+"\t")
        wordsFileF.write("\n");
    wordsFileF.close();


    #===== PRIORS DE LOS TOPICOS Z =====
    zTopicsPrior = numpy.array([aggrMatrixDocsTop.sum(axis=0)])/aggrMatrixDocsTop.shape[0]

    #=====SE COPIA EL CONTENIDO DEL ARCHIVO "FUSIONADO" EN LA MATRIZ EN MEMORIA
    wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/Words_Matrix_Gral.txt","r");
    aggrMatrixTermTop = numpy.zeros((numTerms,topics))

    i=0;
    for elem in wordsFileF:
        lst = list(str(elem).split(" "))
        print lst
        del lst[0];

        for j in range (topics+1):
            if len(lst) <= j:
                break;
            else:
                aggrMatrixTermTop[i][j] = float("{0:.3f}".format(float(lst[j])))

        i = i +1;


    print aggrMatrixTermTop




    #=====SE NORMALIZA LA MATRIZ CREADA PARA QUE TENGA LA FORMA DE UNA PROBABILIDAD CONDICIONAL
    aggrMatrixTermTopGral = aggrMatrixTermTop/numpy.array([aggrMatrixTermTop.sum(axis=1)]).transpose();

    print aggrMatrixTermTopGral

    aggrMatrixTopTermGeneral = generateInverseConditional(aggrMatrixTermTopGral,priorsCorpusGral, TOTAL_TOP_PRIORS_NORM)


    rows = aggrMatrixTermTopGral.shape[0]; cols = aggrMatrixTermTopGral.shape[1]

    wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file_gral.txt","r");
    wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file_gral_normalized.txt","w");
    savedString = "";

    i = 0;
    #=====SE GUARDA LA MATRIZ DE PROBABILIDADES CONDICIONALES=====
    wordList  =[];
    for elem in wordsFileE:
        wordList.append(str(elem).split(" ")[1])

        savedString = savedString + str(elem).split(" ")[1]

        for j in range(cols):
            savedString = savedString + " " + str(float("{0:.3f}".format(float(aggrMatrixTermTopGral[i][j]))));
        i = i + 1;
        wordsFileF.write(savedString+"\n");
        savedString = "";

    wordsFileE.close();
    wordsFileF.close();


    #===== SE CREA UN NUEVEO ARCHIVO CON LOS PSEUDODOCUMENTOS Y LAS PALABRAS
    matrixPeudoDocWords = numpy.matrix(aggrMatrixTermTopGral).transpose();
    #print matrixPeudoDocWords
    rows = matrixPeudoDocWords.shape[0]
    cols = matrixPeudoDocWords.shape[1]
    wordsFileF = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/matrixPeudoDocumentWords.txt","w");

    NUMERODOCUMENTOS = 1;
    for i in range(rows):
        wordsFileF.write("D T:")
        NUMEROPALABRAS = 0;
        for j in range(cols):
            cardinality = int(matrixPeudoDocWords.item(i,j)*10);
            for word in range(cardinality):
                wordsFileF.write(" "+str(wordList[j]));
                NUMEROPALABRAS = NUMEROPALABRAS + 1;

        print NUMERODOCUMENTOS, NUMEROPALABRAS
        if NUMEROPALABRAS == 0:
            wordsFileF.write(allWordsString+"\n")
        else:
            wordsFileF.write("\n")
        NUMERODOCUMENTOS = NUMERODOCUMENTOS + 1;

    wordsFileF.close();


    #===== NOTA (MATRIZ Z W)
    #=====Z, documents
    #=====W, words
    #=====Y, topics
    new_file = file.split("\\")[0]+"/matrixPeudoDocumentWords.txt";
    print "===Inicio LDA para pseudo documento==="
    numTopics  = random.choice(numberOfTopics); num_top_words = 20; alpha = random.choice(alphaChoice); beta = random.choice(betaChoice)
    print numTopics, alpha, beta
    formatingFiles(numTopics,new_file,folder,alpha, beta)
    [matrixDocTerms,matrixProbTermsDoc, matrixDocTermFinal, DTMTrivial, matrixTopWord, matrixDocTop, priorsCorpus, priorsDoc] = LDAPerplexity(numTopics,new_file,folder)

    # matrixDocTop P(Y/Z)
    #print matrixDocTop

    zTopicsPrior = numpy.array([aggrMatrixDocsTop.sum(axis=0)])/aggrMatrixDocsTop.shape[0]
    yTopicsPrior = numpy.array([matrixDocTop.sum(axis=0)])/matrixDocTop.shape[0]

    matAux = numpy.zeros((len(yTopicsPrior[0]),len(zTopicsPrior[0])))

    x = 0; y = 0;
    for elemA in yTopicsPrior[0]:
        for elemB in zTopicsPrior[0]:
            matAux[x][y] = elemB/elemA;
            y = y + 1;
        x = x + 1;
        y = 0;

    matrixTopDoc = matrixDocTop.transpose()*matAux

    #===P(Y/d)
    #=== aggrMatrixDocsTop P(Z/D)
    #=== matrizDocTop P(Y/Z)

    matrixDocTopY = numpy.dot(aggrMatrixDocsTop,matrixDocTop);

    #=== P(W/Y)
    #print matrixTopWord

    #===P(W/D)
    matrixDocWords = numpy.dot(matrixDocTopY,matrixTopWord);
    print "#===P(W/D)"
    #print matrixDocWords

    matAux = numpy.zeros((len(priorsCorpus),len(priorsDocsDGral)))

    x = 0; y = 0;
    for elemA in priorsCorpus:
        for elemB in priorsDocsDGral:
            #print elemA, elemB
            matAux[x][y] = elemB/elemA;
            y = y + 1;
        x = x + 1;
        y = 0;

    print "#===P(D/W)"

    matrixWordDocs = matrixDocWords.transpose()*matAux

    print matrixDocTermsGral; print matrixProbTermsDocGral;

    matrixWordDocs = matrixWordDocs/numpy.array([matrixWordDocs.sum(axis=1)]).transpose()
    print matrixWordDocs



    #print "matrixWordDocs"
    #print matrixWordDocs

    print "Perplexity"
    [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTermsGral,matrixWordDocs,matrixProbTermsDocGral)

    print Perplexity_Prob;
    print Perplexity_Emp;
    print probVectorProb;
    print probVectorEmp;


    print "Perplexity General"
    [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTermsGral,matrixDocTermFinalGral,matrixProbTermsDocGral)
    print Perplexity_Prob;
    print Perplexity_Emp;
    print probVectorProb;
    print probVectorEmp

    from pylab import *
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    img = aggrMatrixTopTermGeneral
    print img
    img = 1- aggrMatrixTopTermGeneral
    print img
    imgplot = plt.imshow(img, interpolation='nearest', vmin = 0, vmax = 1, cmap=mpl.cm.gray)
    plt.grid(True)
    plt.show()



if method == "EXAMPLE":
    import numpy

    matrixDocWordOriginal = numpy.matrix('0.3  0.3  0.3  0.0  0.0  0.0  0.0  0.0  0.0   0.0  0.0  0.0 0.0 0.0 0.0 0.0 0.0;'
                                 '0.2  0.0 0.0  0.2  0.2  0.2 0.2  0.0  0.0   0.0  0.0  0.0 0.0 0.0 0.0 0.0 0.0;'
                                 '0.0  0.0  0.0 0.0  0.0 0.0 0.0 0.3 0.3 0.3 0.0 0.0 0.0 0.0 0.0 0.0 0.0;'
                                 '0.0  0.0  0.0 0.0  0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.25 0.25 0.25 0.0 0.0 0.0;'
                                 '0.0  0.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.2  0.0  0.0 0.0 0.0 0.2 0.2 0.2');




    matrixTopWord = numpy.matrix('0.1  0.1  0.0  0.1  0.0  0.1  0.0  0.1  0.1   0.1  0.0  0.0 0.1 0.1 0.1 0.0 0.0;'
                                 '0.0  0.0  0.14 0.0  0.14 0.0 0.14 0.0 0.0 0.0 0.14 0.14 0.0 0.0 0.0 0.14 0.15');
    print matrixTopWord

    matrixDocTop = numpy.matrix('1.0  0.0 ;'
                                 '1.0  0.0 ;'
                                 '0.0  1.0 ;'
                                 '0.0  1.0 ;'
                                 '1.0  0.0 ');

    matrixDocWord = numpy.matrix('0.1  0.1  0.0  0.1  0.0  0.1  0.0  0.1  0.1   0.1  0.0  0.0 0.1 0.1 0.1 0.0 0.0;'
                                 '0.1  0.1  0.0  0.1  0.0  0.1  0.0  0.1  0.1   0.1  0.0  0.0 0.1 0.1 0.1 0.0 0.0;'
                                 '0.0  0.0  0.14 0.0  0.14 0.0 0.14 0.0 0.0 0.0 0.14 0.14 0.0 0.0 0.0 0.14 0.15;'
                                 '0.0  0.0  0.14 0.0  0.14 0.0 0.14 0.0 0.0 0.0 0.14 0.14 0.0 0.0 0.0 0.14 0.15;'
                                 '0.1  0.1  0.0  0.1  0.0  0.1  0.0  0.1  0.1   0.1  0.0  0.0 0.1 0.1 0.1 0.0 0.0');

    from pylab import *
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    img = matrixDocWordOriginal
    print img
    img = 1- matrixDocWordOriginal
    print img
    imgplot = plt.imshow(img, interpolation='nearest', vmin = 0, vmax = 1, cmap=mpl.cm.gray)
    plt.grid(True)
    plt.show()


    testCorpusPerplexity()



        #== SOLO CON CORPUS COMPLETO ==
        #newsFilesList = ["DS_talk_religion_misc\\talk_religion_misc_train.txt",
        # "DS_talk_politics_misc\\talk_politics_misc_train.txt",
        # "DS_talk_politics_mideast\\talk_politics_mideast_train.txt",
        # "DS_talk_politics_guns\\talk_politics_guns_train.txt",
        # "DS_soc_religion_christian\\soc_religion_christian_train.txt",
        # "DS_sci_space\\sci_space_train.txt",
        # "DS_sci_med\\sci_med_train.txt",
        # "DS_sci_electronics\\sci_electronics_train.txt",
        # "DS_sci_crypt\\sci_crypt_train.txt",
        # "DS_rec_sport_hockey\\rec_sport_hockey_train.txt",
        # "DS_rec_sport_baseball\\rec_sport_baseball_train.txt",
        # "DS_rec_motorcycles\\rec_motorcycles_train.txt",
        # "DS_rec_autos\\rec_autos_train.txt",
        # "DS_misc_forsale\\misc_forsale_train.txt",
        # "DS_comp_windows_x\\comp_windows_x_train.txt",
        # "DS_comp_sys_mac_hardware\\comp_sys_mac_hardware_train.txt",
        # "DS_comp_sys_ibm_pc_hardware\\comp_sys_ibm_pc_hardware_train.txt",
        # "DS_comp_os_ms_windows_misc\\comp_os_ms_windows_misc_train.txt",
        # "DS_comp_graphics\\comp_graphics_train.txt",
        # "DS_alt_atheism\\alt_atheism_train.txt"]
        #topicsList = [16,16,24,20,28,24,18,12,18,18,18,12,18,12,12,12,12,14,16,20]
        #sampleFile =newsFilesList[i];
        #numTopics = topicsList[i]
        #== SOLO CON CORPUS COMPLETO ==





    #
    #     matrixWordTop = matrixTopWord.transpose()
    #
    #     topicPriors = [float(1/float(numTopics))]*numTopics
    #     topicPriors = list(topicPriors)
    #     print matrixWordTop
    #     print "topicPriors====", topicPriors
    #
    #
    #     matrixTopWord = generateInverseConditional(matrixWordTop,priorsCorpus, topicPriors)
    #
    #     saveFileMatrix(matrixTopWord.transpose(),"MATRIXTOPWORDinv",i,file,"BAGGING");
    #     saveFileMatrix(matrixTopWord,"MATRIXTOPWORD",i,file,"BAGGING");
    #     saveFileMatrix(matrixDocTop,"MATRIXDOCTOP",i,file,"BAGGING")
    #     saveFileMatrix(matrixProbTermsDoc,"MATRIXPROBTERMSDOCS",i,file,"BAGGING")
    #     saveFileMatrix(matrixDocTermFinal,"MATRIXDOCTERMSFINAL",i,file,"BAGGING")
    #
    #     saveFileList(priorsCorpus,"PRIORSCORPUS",i,file,"BAGGING")
    #     saveFileList(topicPriors,"TOPICPRIORS",i,file,"BAGGING")
    #
    #
    #     TOTAL_DTMTrivial.append(DTMTrivial)
    #     TOTAL_DTMO.append(matrixDocTerms)
    #     TOTAL_DTMProb.append(matrixProbTermsDoc)
    #     TOTAL_DTMEmp.append(matrixDocTermFinal)
    #     rows = matrixTopWord.transpose().shape[0]; cols = matrixTopWord.transpose().shape[1]
    #     TOTAL_TOP_PRIORS.append(numpy.array([matrixTopWord.sum(axis=1)])/rows);
    #     TOTAL_TOP_WORD_Emp.append(matrixTopWord.transpose())
    #
    #     print "Calculo Perplexity"
    #     [Perplexity_Prob,Perplexity_Emp, probVectorProb, probVectorEmp] = calculatePerplexity(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
    #
    #     perpPerDocFile = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/BAGGING/PerplexityPerDoc"+str(i)+".txt","w");
    #     for l1, l2 in zip(probVectorProb, probVectorEmp):
    #         perpPerDocFile.write(str(l1)+"-"+str(l2)+"="+str(l1-l2)+"\n");
    #     perpPerDocFile.close();
    #
    #
    #     #perpFile.write(str(Perplexity_Prob)+"\t"+str(Perplexity_Emp)+"\t"+str(Perplexity_Prob-Perplexity_Emp)+"\n")
    #
    #     print "Calculo Distancia"
    #     Distance_Emp = calculateDistance(matrixDocTerms,matrixDocTermFinal,matrixProbTermsDoc)
    #     #print Distance_Emp
    #     #distFile.write(str(Distance_Emp)+"\n")
    #
    #     print "it" + str(i)
    #
    # wordsFileE = open("C:/Mallet/testCMD/"+file.split("\\")[0]+"/words_count_file.txt","r");
    # trainWords = [];
    # for elem in wordsFileE: trainWords.append(elem.split(" ")[1])
    # wordsFileE.close()
    #
    #
    # trainPerp = trainPerplexityInv(trainWords, trainWords, testDocs, iterations, file,"BAGGING")
    #
    # testPerp = trainPerplexityInv(trainWords, testWords, testDocs, iterations, file,"BAGGING")
    #
    #
    # #---import sys; sys.exit("Error message")
    #
    # MEAN_TOTAL_DTMProb = sum(TOTAL_DTMProb)/iterations
    # MEAN_TOTAL_DTMEmp = sum(TOTAL_DTMEmp)/iterations
    #
    # print iterations
    # #import sys; sys.exit("Error message")
    # [Perplexity_Prob_Final, Perplexity_Emp_Final, probVectorProbFinal, probVectorEmpFinal] = calculatePerplexity(sum(TOTAL_DTMO)/iterations,sum(TOTAL_DTMEmp)/iterations,sum(TOTAL_DTMProb)/iterations)
    #
    # print "Perplexity_Prob_Final, Perplexity_Emp_Final"
    # print Perplexity_Prob_Final, Perplexity_Emp_Final
    #
    # #perpFile.write("FINAL: "+str(Perplexity_Prob_Final)+"\t"+str(Perplexity_Emp_Final)+"\t"+str(Perplexity_Prob_Final-Perplexity_Emp_Final)+"\n")
    #
    # Distance_Emp_Final = calculateDistance(sum(TOTAL_DTMO)/iterations,sum(TOTAL_DTMEmp)/iterations,sum(TOTAL_DTMProb)/iterations)
    #
    # print "Distance_Emp_Final"
    # print Distance_Emp_Final
    #
    #
    #
    # matrixDocTermEmp_plot = sum(TOTAL_DTMEmp)/iterations;
    # matrixDocTermProb_plot = sum(TOTAL_DTMProb)/iterations
    # matrixTopWords_plot = sum(TOTAL_TOP_WORD_Emp)/iterations;
    #
    #
    #
    # kld = KLD(MEAN_TOTAL_DTMEmp, MEAN_TOTAL_DTMProb)
    #
    # print trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld
    # return trainPerp, testPerp, Perplexity_Emp_Final, Distance_Emp_Final, kld
    #




