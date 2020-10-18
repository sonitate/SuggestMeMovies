from rake_nltk import Rake


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
  r = Rake()
  r.extract_keywords_from_text(text)
  A = r.get_ranked_phrases()
  B = r.get_ranked_phrases_with_scores()
  print("*** Ranked phrases ***\n\n")
  print(A)
  print("****************************************************************\n\n\n*** Ranked phrases with scores ***\n\n")
  print(B)

plots = "CleanData/plot_summaries.txt"
plotList = getPlots(plots)

for plot in plotList[:5]:
  print(plot[0])
  print('\n')
  getKeywords(plot[1])
  print('\n\n****************************************************************\n\n')
