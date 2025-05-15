import csv
import os
from nltk import TreebankWordDetokenizer

#write script mods to calculate summary token length and write to new files
def process(input_file, output_file, wordcount_file):
    #detokenizer=TreebankWordDetokenizer()  #detokenizing text only

    with open(input_file, 'r', encoding='utf-8-sig') as inpFile:
        file_reader=csv.reader(inpFile, delimiter=',')
        with open(output_file, 'w', encoding='utf-8') as outFile:
            with open(wordcount_file, 'w') as wordOut:
                k=0
                truncations=0
                for item in file_reader:
                    if k==0:    #skip header
                        k+=1
                        continue
                    else:       #write only summaries to output file
                        #print(item[5])
                        wordOut.write(str(k) + " Count: " + str(len(item[5])) +'\n')
                        if item[5][-1] != '.':
                            wordOut.write("Warning: No period detected. Likely Truncated. \n")
                            truncations+=1
                            i=-1
                            #handling truncation
                            while item[5][i]!='.':
                                i += -1
                            #commented code for detokenizing text only 
                            '''summary=detokenizer.detokenize(item[5][:i+1].split()) 
                            summary=summary.replace(' .', '.')
                            outFile.write(summary+ '\n') '''
                            outFile.write(item[5][:i+1]+'\n')   #write tokenized text to output file

                        else:
                            #commented code for detokenizing text only
                            '''summary=detokenizer.detokenize(item[5].split())
                            summary=summary.replace(' .', '.')
                            outFile.write(summary+ '\n')'''
                            outFile.write(item[5] + '\n')  #write tokenized text to output file
                        k+=1
                wordOut.write("Total number of truncations: " +str(truncations))
            wordOut.close()
        outFile.close()
    inpFile.close()

#checks for truncation in tokenized text only
def truncation_check(input_file):
    with open(input_file, 'r', encoding="utf-8") as inpFile:
        k=1
        for item in inpFile:
            if item[-2] != '.': #print warning to terminal if still truncated
                print(item[-2])
                print("Line {} - Warning: No period detected. Still Truncated.".format(k))
            k+=1
    inpFile.close()
            
                
def main():
     #read in csv files with predictions
     plos_input='plos_predictions.csv'
     elife_input="elife_predictions.csv"

     plos_out="plos.txt" #proper text formatting for submission
     elife_out="elife.txt" #proper text formatting for submission

     #word count files and truncation warnings pre-processing - total truncations listed at end
     plos_words="plos_wordcount.txt"
     elife_words="elife_wordcount.txt"

     #process predictions into proper codabench output files
     process(plos_input, plos_out, plos_words)
     process(elife_input, elife_out, elife_words)

     #check for any remaining truncation in tokenized text
     truncation_check(plos_out)
     truncation_check(elife_out)
       

if __name__ == "__main__":
    main()