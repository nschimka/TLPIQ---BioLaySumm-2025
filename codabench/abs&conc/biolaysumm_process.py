import csv

#write to new files
def process(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8-sig') as inpFile:
        file_reader=csv.reader(inpFile, delimiter=',')
        with open(output_file, 'w', encoding='utf-8') as outFile:
            k=0
            for item in file_reader:
                if k==0:    #skip header
                    k+=1
                    continue
                else:       #write only summaries to output file                                                        
                    outFile.write(item[5] + '\n')  #write tokenized text to output file
                    k+=1
        outFile.close()
    inpFile.close()
                
def main():
     #read in csv files with predictions
     plos_input='plos_predictions.csv'
     elife_input="elife_predictions.csv"

     plos_out="plos.txt" #proper text formatting for submission
     elife_out="elife.txt" #proper text formatting for submission

     #process predictions into proper codabench output files
     process(plos_input, plos_out)
     process(elife_input, elife_out)
       

if __name__ == "__main__":
    main()