import numpy as np
import scipy as sp
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
from matplotlib import rc
from lmfit import models
from lmfit import Model
from ipywidgets import interactive

def fullproduct(a,b):
    aa = np.repeat(a,len(b))
    bb = np.concatenate(np.repeat([b],len(a),axis=0))
    return np.array([aa,bb])

def tripleproduct(a,b,c):
    aa = np.repeat(a,len(b)*len(c))
    bb = np.concatenate(np.repeat([np.repeat(b,len(c))],len(a),axis=0))
    cc = np.concatenate(np.repeat([c],len(a)*len(b),axis=0))
    return np.array([aa,bb,cc])

def f_mkrverts(x,y,xlim=[],ylim=[]):
    if len(xlim)==0:
        xlim = x[[0,-1]]
    if len(ylim)==0:
        ylim = y[[0,-1]]
    xcond = (x>=xlim[0])&(x<=xlim[1])
    xlen = np.sum(xcond)
    mkrwidth = 1/xlen
    ycond = (y>=ylim[0])&(y<=ylim[1])
    ylen = np.sum(ycond)
    mkrheight = 1/ylen
    mkrverts = list(zip(np.array([-mkrwidth,mkrwidth,mkrwidth,-mkrwidth]),np.array([-mkrheight,-mkrheight,mkrheight,mkrheight])))
    return mkrverts,xlen,ylen
    

def trsubplot(x,y,nrows=2,ncols=1,index = [0],xlim = 'auto',ylim='auto',xlabel='',ylabel='',xticks=[],xticklabels=[],off = 0,plotsize=[34,10],markersize=2,linewidth=0.5,marker='o',linestyle='-',font1=30,font2=20,title=[],label = [],labelcolor = 'dodgerblue',colors='gist_rainbow',yfactor=1,yoff=0,markerscale=2,legend='on'):
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    #clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    if x =='auto':
        x = {}
        for i in range(len(y)):
            x[i] = {}
            for j in range(len(y[i])):
                x[i][j] = np.arange(len(y[i][j]))
    if len(title) == 0:
        title = {}
        for i in range(nrows*ncols):
            title[i] = ''
    if len(label) == 0:
        label = np.repeat('',len(x[0]))
    if len(np.shape(label)) == 1:
        label = np.repeat([label],len(x),axis=0)
    if len(np.shape(ylim)) == 1:
        ylim = np.repeat([ylim],nrows*ncols,axis=0)
    if len(np.shape(markersize))==0:
        markersize = np.repeat(markersize,len(x[0]))
    if len(np.shape(markersize))==1:
        markersize = np.repeat([markersize],len(x),axis=0)
    if len(np.shape(linestyle))==0:
        linestyle = np.repeat(linestyle,len(x[0]))
    if len(np.shape(linestyle))==1:
        linestyle = np.repeat([linestyle],len(x),axis=0)
    if len(np.shape(ylabel))==0:
        ylabel = np.repeat([ylabel],len(x),axis=0)
    if len(np.shape(xlabel))==0:
        xlabel = np.repeat([xlabel],len(x),axis=0)
    for i in range(nrows*ncols):
        ax = fig.add_subplot(nrows,ncols,i+1)
        colors = cm.prism(np.linspace(0, 1, len(index)))
        #if colors == 'prism':
        #    colors = cm.prism(np.linspace(0, 1, len(index)+1))
        #if colors == 'gist_rainbow':
        #    colors = cm.gist_rainbow(np.linspace(0, 1, len(index)+1))
        #if colors == 'inferno':
        #    colors = cm.inferno(np.linspace(0, 1, len(index)+1))
        for l,j in enumerate(index):
            #ind = np.min([len(x[i][j]),len(y[i][j])])
            ax.plot(x[i][j], y[i][j]*yfactor+yoff*l, color=colors[l],ms=markersize[i][j],marker=marker,label = label[l],linestyle=linestyle[i][j],linewidth=linewidth)
        ax.set_ylabel(ylabel[i], color=labelcolor,fontsize = font1)
        ax.set_xlabel(xlabel[i], color=labelcolor,fontsize = font1)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
        ax.set_title(title[i],fontsize=font1,color=labelcolor)
        if legend != 'off':
            ax.legend(fontsize = font2,markerscale = markerscale)
        if xlim != 'auto':
            ax.set_xlim([xlim[0],xlim[1]])
        if ylim != 'auto':
            ax.set_ylim([ylim[i][0],ylim[i][1]])
        if len(xticks) != 0:
            ax.set_xticks(xticks)
        if len(xticklabels) != 0:
            ax.set_xticklabels(xticklabels)
    plt.show()
def trplot(x,y,index = [0],xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[1,1.2],markersize=2,markerscale = 5,linewidth=0.5,marker='o',linestyle='-',font1=30,font2=20,title='',label = '',colors='',save=False,savename='no_name'):
    plt.rcParams['axes.facecolor']='white'
    try:
        np.shape(x)
    except: # this is handeling an error that comes when having x as list of different length arrays. np.shape throws error for that "inhomogenous ..."
        x = dict(zip(np.arange(len(x)),x))
        y = dict(zip(np.arange(len(y)),y))
    if len(np.shape(x))==1:
        x = np.array([x])
        y = np.array([y])
    if len(np.shape(markersize)) == 0:
        markersize = np.repeat([markersize],len(x))
    if len(np.shape(label)) == 0:
        label = np.repeat([label],len(x))
    if len(np.shape(linestyle)) == 0:
        linestyle = np.repeat([linestyle],len(x))
    if len(np.shape(off)) == 0:
        off = np.repeat([off],len(x))
    if len(np.shape(marker)) == 0:
        marker = np.repeat([marker],len(x))
    if len(np.shape(linewidth)) == 0:
        linewidth = np.repeat([linewidth],len(x))
    if len(colors) == 1:
        colors = np.repeat(colors,len(x))
    elif len(colors) == 0:
        colors = cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    for i in index:
        ax.plot(x[i],y[i]+i*off[i],marker=marker[i],color=colors[i],markersize = markersize[i],linewidth=linewidth[i],label = label[i],linestyle = linestyle[i])
    ax.set_ylabel(ylabel, color=labelcolor,fontsize = font1)
    ax.set_xlabel(xlabel, color=labelcolor,fontsize = font1)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    if label[0] != '':
        ax.legend(fontsize = font2,markerscale = markerscale)
    ax.set_title(title,fontsize=font2,color=labelcolor)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])    
    if xlim != 'auto':
        ax.set_xlim([xlim[0],xlim[1]])
    if save:
        plt.savefig(savename,format=savename[-3:],bbox_inches='tight')
    plt.show()
    
def errorbar(x,y,err,colors=0,off=0,index = [0],xlim = '',ylim='',xlabel='',ylabel='',
             labelcolor = 'black',plotsize=[1,1.2],markersize=2,linewidth=0.5,marker='o',
             linestyle='-',font1=30,font2=20,title='',label = '',save=False,savename='no_name.jpg',
             mfc='',mew=.5,ecolor=0,elinewidth=.5,tw=1,tl=1,tml=.5,xtickmul=1000,ytickmul=1000,savefile='',
             yscale = 'linear', **kwargs):
    plt.rcParams['axes.facecolor']='white'
    if len(np.shape(markersize)) == 0:
        markersize = np.repeat([markersize],len(x))
    if len(np.shape(marker)) == 0:
        marker = np.repeat([marker],len(x))
    if len(np.shape(label)) == 0:
        label = np.repeat([label],len(x))
    if len(np.shape(linestyle)) == 0:
        linestyle = np.repeat([linestyle],len(x))
    if len(np.shape(linewidth)) == 0:
        linewidth = np.repeat([linewidth],len(x))
    if len(np.shape(elinewidth)) == 0:
        elinewidth = np.repeat([elinewidth],len(x))
    if len(np.shape(off)) == 0:
        off = np.repeat([off],len(x))
    if len(np.shape(colors)) == 0:
        colors = cm.rainbow(np.linspace(0, 1, len(x)))
    if len(np.shape(ecolor)) == 0:
        try: 
            ecolor = colors
        except:
            ecolor = cm.rainbow(np.linspace(0, 1, len(x)))
    if len(np.shape(mfc)) == 0:
        mfc = colors
    if len(np.shape(mew)) == 0:
        mew = np.repeat([mew],len(x))
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    #print(colors)
    for iii in index:
        ax.errorbar(np.array(x[iii],dtype=float), np.array(y[iii],dtype=float)+np.array(off[iii], dtype=float), np.array(err[iii],dtype=float), marker=marker[iii], color=colors[iii], markersize = markersize[iii], linewidth=linewidth[iii], label = label[iii], linestyle = linestyle[iii], mfc=mfc[iii], mew=mew[iii], ecolor=ecolor[iii],elinewidth=elinewidth[iii])
    ax.set_ylabel(ylabel, color=labelcolor,fontsize = font1)
    ax.set_xlabel(xlabel, color=labelcolor,fontsize = font1)
    #ax.secondary_yaxis('right',functions=(lambda x: x/2-.3, lambda x: (x+.3)*2),yticks = [1,2,3])
    ax.xaxis.set_minor_locator(MultipleLocator(xtickmul))
    ax.yaxis.set_minor_locator(MultipleLocator(ytickmul))
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2,right=True,which='both',direction='in',width=tw,length=tl)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=font2,top=True,which='both',direction='in',width=tw,length=tl)
    ax.tick_params(axis = 'both',which='minor',length=tml)
    if yscale == 'log':
        ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    #remove the errorbars
    handles = [h[0] for h in handles]
    if label[0] != '':
        ax.legend(handles,labels,fontsize = font2,markerscale = 2)
    ax.set_title(title,fontsize=font2,color=labelcolor)
    if len(ylim) != 0:
        ax.set_ylim([ylim[0],ylim[1]])    
    if len(xlim) != 0:
        ax.set_xlim([xlim[0],xlim[1]])
    if save:
        plt.savefig(savename,format=savename[-3:],bbox_inches='tight')
    plt.show()
    return fig, ax

def plotvol(M,a,b,c,d,color='inferno',labelcolor='black',labelsize1=20,labelsize2=30,xlabel='',ylabel='',zlabel='',climsliders = [0,3,1,5]):
    color = 'inferno'
    h=M[0];k=M[1];E=M[2];I=M[3];
    def pt(a,b,c,d):
        fig = plt.figure()
        plotsize=[2,2]
        clim = [c,d]
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
        img = ax.scatter(h[(I>clim[0])&(I<clim[1])], k[(I>clim[0])&(I<clim[1])], E[(I>clim[0])&(I<clim[1])], c=I[(I>clim[0])&(I<clim[1])], cmap=color)
        fig.colorbar(img)
        ax.set_xlabel(xlabel, color=labelcolor,fontsize = labelsize2)
        ax.set_ylabel(ylabel, color=labelcolor,fontsize = labelsize2)
        ax.set_zlabel(zlabel, color=labelcolor,fontsize = labelsize2)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=labelsize1)
        ax.tick_params(axis = 'z',labelcolor=labelcolor,color=labelcolor,labelsize=labelsize1)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=labelsize1)
        ax.view_init(a,b)
        #fig.colorbar
        #img.set_clim(0.05,.1)
        #ax.legend(fontsize = 14,markerscale = 5)
        #ax.set_xlim([xlim[0],xlim[1]])
        #ax.set_title('offset = ' + str(off),fontsize=20,color=labelcolor)
        plt.show()
def pt(M,model,a,b,c=1,d=3,color='inferno',plotsize=[34,10],l = 100,markersize=5,marker='o'):
    u=np.array([M[1],model[1]]);v=np.array([M[2],model[2]]);E=np.array([M[3],model[3]]);Ia=np.array([M[4],model[4]]);
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    labelcolor = 'dodgerblue'
    for i in range(2):
        ax = fig.add_subplot(1,2,i+1, projection='3d')
        img = ax.scatter(u[i][(Ia[i]>clim[0])&(Ia[i]<clim[1])], v[i][(Ia[i]>clim[0])&(Ia[i]<clim[1])], E[i][(Ia[i]>clim[0])&(Ia[i]<clim[1])], c=Ia[i][(Ia[i]>clim[0])&(Ia[i]<clim[1])], cmap=color,s=markersize,marker=marker)
        fig.colorbar(img)
        ax.set_ylabel('k', color=labelcolor,fontsize = 30)
        ax.set_xlabel('h', color=labelcolor,fontsize = 30)
        ax.set_zlabel('E', color=labelcolor,fontsize = 30)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.tick_params(axis = 'z',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.view_init(a,b)
        ax.set_zlim(0,1.3)
        #fig.colorbar
        #img.set_clim(0.05,.1)
        #ax.legend(fontsize = 14,markerscale = 5)
        #ax.set_xlim([xlim[0],xlim[1]])
        #ax.set_title('offset = ' + str(off),fontsize=20,color=labelcolor)
    plt.show()
def pt2(M,color='inferno',plotsize=[34,10],markersize=5,marker='o',clim = '',title = ''):
    u=M[0];v=M[1];Ia=M[2];
    if clim == '':
        clim = [0,np.max(Ia)]
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    #clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    labelcolor = 'black'
    ax = fig.add_subplot(1,2,1)
    img = ax.scatter(u, v, c=Ia, cmap=color,s=markersize,marker=marker)
    fig.colorbar(img)
    ax.set_ylabel('k', color=labelcolor,fontsize = 30)
    ax.set_xlabel('h', color=labelcolor,fontsize = 30)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    img.set_clim(clim[0],clim[1])
    #fig.colorbar
    #ax.legend(fontsize = 14,markerscale = 5)
    #ax.set_xlim([xlim[0],xlim[1]])
    ax.set_title(title,fontsize=20,color=labelcolor)
    plt.show()
def pt2sub(M,color='',plotsize=[34,10],l = 100,markersize=5,marker='o',nrows=1,ncols=1,clim = '',title='',ylim = '',xlim = '',clabel='',ylabel = '',xlabel='',labelcolor='black',cfontsize = 50,fonts1=30,fonts2=30,save=False,savefile='',dpi=600): #plot, 2 dimensional, with subtitles
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    #clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')'
    if title == '':
        title = {}
        for i in range(nrows*ncols):
            title[i] = ''
    if clim == '':
        clim = {}
        for i in range(nrows*ncols):
            clim[i] = [0,np.max(M[i][2])]
    if color == '':
        color = {}
        for i in range(nrows*ncols):
            color[i] = 'inferno'
    if len(np.shape(markersize))==0:
        a = markersize
        markersize = {}
        for i in range(nrows*ncols):
            markersize[i] = a
    for i in range(nrows*ncols):
        ax = fig.add_subplot(nrows,ncols,i+1)
        img = ax.scatter(M[i][0], M[i][1], c=M[i][2], cmap=color[i],s=markersize[i],marker=marker)
        cbar = fig.colorbar(img)
        ax.set_ylabel(ylabel, color=labelcolor,fontsize = fonts2)
        ax.set_xlabel(xlabel, color=labelcolor,fontsize = fonts2)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fonts1)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=fonts1)
        cbar.set_label(clabel,rotation=-90,fontsize=cfontsize)
        cbar.ax.tick_params(labelsize=fonts1)
        img.set_clim(clim[i][0],clim[i][1])
        ax.set_title(title[i],fontsize=fonts1,color=labelcolor)
    #ax.legend(fontsize = 14,markerscale = 5)
        if xlim != '':
            ax.set_xlim([xlim[0],xlim[1]])
        if ylim != '':
            ax.set_ylim([ylim[0],ylim[1]])
    plt.show()
    if save:
        fig.savefig(savefile,dpi=dpi)
def plot2d(x,y,index = [0],xlim = [0,1.4],ylim='auto',off = 0,labelcolor = 'black',plotsize=[2.1,2.5]):
    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    for i in index:
        ax.plot(data[i][0],data[i][1]+i*off,'-o',color=colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = str(fields[i])+' T, i = ' + str(i))
    ax.set_ylabel('I', color=labelcolor,fontsize = 30)
    ax.set_xlabel('E (meV)', color=labelcolor,fontsize = 30)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    ax.legend(fontsize = 14,markerscale = 5)
    ax.set_xlim([xlim[0],xlim[1]])
    ax.set_title('offset = ' + str(off),fontsize=20,color=labelcolor)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])
def plothist(h=[[]],aspect=1,clim='auto',plotsize=[2.1,2.5],nrows=1,ncols=1,index = [0],xlim = 'auto',ylim='auto',xlabel='',ylabel='',markersize=2,linewidth=0.5,marker='o',linestyle='-',font1=30,font2=20,title='',labelcolor='black',c='gist_rainbow',yfactor=1,yoff=0,markerscale=2,numplots = 1,shrink=1):
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    #clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    if title == '':
        title = {}
        for i in range(nrows*ncols):
            title[i] = ''
    if len(np.shape(ylabel))==0:
        ylabel = np.repeat([ylabel],numplots,axis=0)
    if len(np.shape(xlabel))==0:
        xlabel = np.repeat([xlabel],numplots,axis=0)
    for i in range(numplots):
        ax = fig.add_subplot(nrows,ncols,i+1)
        im = ax.imshow(h[i],cmap=c,aspect = aspect,interpolation='none',origin='lower')
        ax.set_ylabel(ylabel[i], color=labelcolor,fontsize = 30)
        ax.set_xlabel(xlabel[i], color=labelcolor,fontsize = 30)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.set_title(title[i],fontsize=20,color=labelcolor)
        ax.legend(fontsize = 14,markerscale = markerscale)
        cbar = fig.colorbar(im,shrink = shrink)
        if xlim != 'auto':
            ax.set_xlim([xlim[0],xlim[1]])
        if ylim != 'auto':
            ax.set_ylim([ylim[0],ylim[1]])
        if clim != 'auto':
            im.set_clim(clim[0],clim[1])
        plt.show()
        plt.rcParams['axes.facecolor']='white'
#    fig = plt.figure()
#    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
#    im = ax.imshow(h,cmap='rainbow',aspect = aspect,interpolation='none',origin='low')  
#    ax.set_xlabel(xlabel,fontsize = 20)
#    ax.set_ylabel(ylabel,fontsize = 20)
#    #ax.set_axis(fontsize = 20)
#    ax.set_title(title)
#    cbar = fig.colorbar(im,shrink = 0.8)
#    if ylim != 'auto':
#        ax.set_ylim([ylim[0],ylim[1]])    
#    if xlim != 'auto':
#        ax.set_xlim([xlim[0],xlim[1]])
#    if clim != 'auto':
#        im.set_clim(clim[0],clim[1])
#    plt.show()
def pcolormesh(x = [[]], y = [[]], c=[[]],title='',aspect=1,index = [0],clim=[],xlim = [],ylim=[],norm='linear',
               xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5],clabel = '',cbarshrink = 1,
               font1=30,font2=20,cmap = 'gnuplot2',tw=1,tmw=1,tl=1,tml=.5,xtickmul=100000,ytickmul=100000,save=False,
               savename='no_name.jpg',xticks=[],yticks=[]): # x and y can also be one dimensional, matching dimensions of c (+1 if using shading='flat')
    plt.rcParams['axes.facecolor']='white'
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    if norm == 'linear':
        norm = mpl.colors.Normalize()
    elif norm == 'log':
        # norm=colors.LogNorm(vmin=np.min(c), vmax=np.max(c))
        norm=mpl.colors.LogNorm()
    im = ax.pcolor(x,y,c,cmap=cmap,shading='nearest',norm=norm)#,aspect = aspect,interpolation='none',origin='low')  
    ax.set_xlabel(xlabel,fontsize = font1,color=labelcolor)
    ax.set_ylabel(ylabel,fontsize = font1,color=labelcolor)
    #ax.set_axis(fontsize = 20)
    ax.set_title(title)    
    if len(xticks)!=0:
        ax.set_xticks(xticks)
    if len(yticks)!=0:
        ax.set_xticks(yticks)
    ax.xaxis.set_minor_locator(MultipleLocator(xtickmul))
    ax.yaxis.set_minor_locator(MultipleLocator(ytickmul))
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2,right=True,which='both',direction='in',width=tw,length=tl)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=font2,top=True,which='both',direction='in',width=tw,length=tl)
    ax.tick_params(axis = 'both',which='minor',length=tml,width=tmw)
    cbar = fig.colorbar(im,shrink = cbarshrink)
    cbar.ax.set_ylabel(clabel,fontsize = font1)
    cbar.ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    if len(ylim) != 0:
        ax.set_ylim([ylim[0],ylim[1]])    
    if len(xlim) != 0:
        ax.set_xlim([xlim[0],xlim[1]])
    if len(clim) != 0:
        im.set_clim(clim[0],clim[1])
    if save:
        plt.savefig(savename,format=savename[-3:],bbox_inches='tight')
    plt.show()
def scatter2d(x = [], y = [], c=[],title='',aspect=1,index = [0],clim=[],xlim = [],ylim=[],norm='linear',
               xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5],clabel = '',cbarshrink = 1,
               font1=30,font2=20,cmap = 'gnuplot2',tw=1,tmw=1,tl=1,tml=.5,xtickmul=100000,ytickmul=100000,save=False,
               savename='no_name.jpg',xticks=[],yticks=[],marker='s',markersize=2):
    plt.rcParams['axes.facecolor']='white'
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    if norm == 'linear':
        norm = mpl.colors.Normalize()
    elif norm == 'log':
        # norm=colors.LogNorm(vmin=np.min(c), vmax=np.max(c))
        norm = mpl.colors.LogNorm()
    im = ax.scatter(x,y,c=c,cmap=cmap,norm=norm,marker=marker,s=markersize)#,aspect = aspect,interpolation='none',origin='low')  
    ax.set_xlabel(xlabel,fontsize = font1,color=labelcolor)
    ax.set_ylabel(ylabel,fontsize = font1,color=labelcolor)
    #ax.set_axis(fontsize = 20)
    ax.set_title(title,fontsize=font1)    
    if len(xticks)!=0:
        ax.set_xticks(xticks)
    if len(yticks)!=0:
        ax.set_xticks(yticks)
    ax.xaxis.set_minor_locator(MultipleLocator(xtickmul))
    ax.yaxis.set_minor_locator(MultipleLocator(ytickmul))
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2,right=True,which='both',direction='in',width=tw,length=tl)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=font2,top=True,which='both',direction='in',width=tw,length=tl)
    ax.tick_params(axis = 'both',which='minor',length=tml,width=tmw)
    cbar = fig.colorbar(im,shrink = cbarshrink)
    cbar.ax.set_ylabel(clabel,fontsize = font1)
    cbar.ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    if len(ylim) != 0:
        ax.set_ylim([ylim[0],ylim[1]])    
    if len(xlim) != 0:
        ax.set_xlim([xlim[0],xlim[1]])
    if len(clim) != 0:
        im.set_clim(clim[0],clim[1])
    if save:
        plt.savefig(savename,format=savename[-3:],bbox_inches='tight')
    plt.show()
def f_custommarker(x,y):
    mkrwidth= np.abs(np.average(x[1:]-x[:-1]))*1000
    mkrheight= np.abs(np.average(y[1:]-y[:-1]))*1000
    mkrverts = list(zip(np.array([-mkrheight,mkrheight,mkrheight,-mkrheight])/2,np.array([-mkrwidth,-mkrwidth,mkrwidth,mkrwidth])/2))
    return mkrverts
def pcolormeshsub(x = [], y = [], c=[[]],title='',aspect=1,index = [0],clim='auto',xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5],clabel = '',cbarshrink = 1,fonts1=30,fonts2=20,cmap = 'gnuplot2'):
    plt.rcParams['axes.facecolor']='white'
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    im = ax.pcolormesh(x,y,c,cmap=cmap,shading='flat')#,aspect = aspect,interpolation='none',origin='low')  
    ax.set_xlabel(xlabel,fontsize = fonts1,color=labelcolor)
    ax.set_ylabel(ylabel,fontsize = fonts1,color=labelcolor)
    #ax.set_axis(fontsize = 20)
    ax.set_title(title)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    cbar = fig.colorbar(im,shrink = cbarshrink)
    cbar.ax.set_ylabel(clabel,fontsize = fonts1)
    cbar.ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])    
    if xlim != 'auto':
        ax.set_xlim([xlim[0],xlim[1]])
    if clim != 'auto':
        im.set_clim(clim[0],clim[1])
    plt.show()
def trfit2G(x,y,hints):
    model = models.GaussianModel(prefix='one_')+models.GaussianModel(prefix='two_')
    parnames = model.param_names
    pars = model.make_params()
    for j,n in enumerate(parnames):
        pars[n].set(value = hints[j],vary=True)
    result = model.fit(y,pars,x=x)
    print(result.fit_report())
    return result
#def oplotsim(self,index,off):
#    for i in index:
#        self.ax.plot(self.x,self.sim[i]+i*off,'-',color=self.colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = 'sim, i = ' + str(i))
#def oplotfit(self,index,off):
#    for i in index:
#        self.ax.plot(self.x,self.simfit[i]+i*off,'-',color=self.colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = 'sim, i = ' + str(i))
