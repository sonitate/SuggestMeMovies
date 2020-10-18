import spacy
import pytextrank



def getPlots(plots):
  f = open(plots, 'r')
  lines = f.readlines()
  f.close()
  plotList = []
  for line in lines:
    line = line.strip()
    plot = line.split('\t')
    #if len(plot) != 2:
      #print("Not 2 elements but " + str(len(plot)))
    plotList.append(plot)

  return plotList

def getKeywords(text):

  # load a spaCy model, depending on language, scale, etc.
  nlp = spacy.load("en_core_web_sm")

  # add PyTextRank to the spaCy pipeline
  tr = pytextrank.TextRank()
  nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

  doc = nlp(text)

  # examine the top-ranked phrases in the document
  for p in doc._.phrases:
    print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
    print(p.chunks)


plots = "CleanData/plot_summaries.txt"
plotList = getPlots(plots)

for plot in plotList[:5]:
  print(plot[0])
  print('\n')
  getKeywords(plot[1])
  print('\n\n****************************************************************\n\n')
