import math
import numpy as np
import matplotlib.pyplot as plt

def accuracy(truePosi, trueNega, falsePosi, falseNega): # Count of all four
	return (truePosi+trueNega)/(truePosi+trueNega+falseNega+falsePosi)

def precision(truePosi, trueNega, falsePosi, falseNega):
	if (truePosi+falsePosi) == 0:
		return 0
	preposi = truePosi/(truePosi+falsePosi)
	# prenega = trueNega/(trueNega+falseNega)
	return preposi

def recall(truePosi, trueNega, falsePosi, falseNega):
	if (truePosi+falseNega)== 0:
		return 0
	recposi = truePosi/(truePosi+falseNega)
	# recnega = trueNega/(trueNega+falsePosi)
	return recposi

def fscore(truePosi, trueNega, falsePosi, falseNega, beta: 1):
	pre = precision(truePosi, trueNega, falsePosi, falseNega)
	rec = recall(truePosi, trueNega, falsePosi, falseNega)
	if (pre*(beta**2)+rec) == 0:
		return 0
	f = (1+beta**2)*((pre*rec)/(pre*(beta**2)+rec))
	return f

def markdowntemplate(tp,tn,fp,fn,beta,title):
	acc = accuracy(tp,tn,fp,fn)
	pre = precision(tp,tn,fp,fn)
	rec = recall(tp,tn,fp,fn)
	fsc = fscore(tp,tn,fp,fn,beta)
	return

def confusionmatrix(truePosi, trueNega, falsePosi, falseNega, title=""):
	fig = plt.figure()
	plt.title(title)
	col_labels = ['Predict:+', 'Predict:-']
	row_labels = ['Real:+', 'Real:-']
	table_vals = [[truePosi, falseNega], [falsePosi, trueNega]]
	the_table = plt.table(cellText=table_vals,
                      colWidths=[0.1] * 3,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(24)
	the_table.scale(4, 4)
	plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
	plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)

	for pos in ['right','top','bottom','left']:
		plt.gca().spines[pos].set_visible(False)

	plt.show()	
	return 
