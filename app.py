# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 21:07:24 2022

@author Guy
"""

from flask import Flask, render_template, redirect, request, url_for, send_from_directory, session
import os
from werkzeug.utils import secure_filename

#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.mixture import GaussianMixture

global data


#%matplotlib inline

app = Flask(__name__)
app.secret_key= "this is customer segmentation app"


app.config['DIRECTORY']= 'static/uploads_datasets/'
app.config['IMG_DIRECTORY']= 'static/images/segments/whole_dataset/'
app.config['IMG_DIRECTORY_BAR']= 'static/images/segments/whole_dataset_bar/'
app.config['IMG_DIRECTORY_VARS']= 'static/images/segments/variables/'
app.config['IMG_DIRECTORY_VARS_GMM']= 'static/images/segments/variables_gmm/'
app.config['IMG_DIRECTORY_ELBOWS']= 'static/images/elbow_graph/'
app.config['DOCUMENTATION'] = 'static/documents/documentation/'
message=''

@app.route('/')
def index():
    session["dataset"]= 'Inactive'
    status= session["dataset"]
    
    return render_template('index.html', status= status)

@app.route('/loaddataset/', methods= ['POST', 'GET'])
def load_dataset():
    if request.method=="POST":
        file = request.files['dataset']
        #save_as= 'dataset'
        if file:
            file.save(os.path.join(
                app.config['DIRECTORY'],
                secure_filename(file.filename)
            ))
            message= 'dataset uploaded successfully'
            #file.save(f'static/uploads_datasets/{file.filename}')
            
            ###EXTRACTING THE DATASET
            global dataset
            try:
                dataset= pd.read_csv(app.config['DIRECTORY']+ secure_filename(file.filename))
                
                dataset_header= dataset.head()
                dataset_tail= dataset.tail()
                dataset_info= dataset.describe()
                dataset_corr= dataset.corr()
                dfname= file.filename
                dfshape= dataset.shape
                dfsize= dataset.size
                dfcolumns= dataset.columns
                dataset_nulls= dataset.isnull().sum()
                null_columns= dataset.isnull()
                global data
                data= dataset
                message= 'Done! Please check the information about the dataset below!>'
                session["dataset"]= 'active'
                status = session["dataset"]
                
                heady= dataset.head()
                
                return render_template('index.html', heady=heady.to_html(), status=status, message=message,null_columns=null_columns,dataset_nulls= dataset_nulls, dataset_header= dataset_header.to_html(), dataset_tail=dataset_tail.to_html(), dataset_info= dataset_info.to_html(),dataset_corr=dataset_corr.to_html(),dfname=dfname,dfshape=dfshape, dfsize=dfsize, dfcolumns=dfcolumns  )
           
            except Exception as e:
                message = e
                print('Error: ', e)
            finally:
                print('Finished excecution!')
                
            
            
        
        return redirect('/')
    
    return redirect('/')

@app.route('/preprocess.html/', methods=['POST', 'GET'])
def processing():
    try:
        dtypes= data.dtypes
        columns= data.columns
        dataset_nulls= data.isnull().sum()
        return render_template('preprocess.html',dtypes=dtypes, dataset_nulls=dataset_nulls, data= dataset, columns=columns)
    except Exception as e:
        return render_template("index.html", message= "PLease load the dataset first!!")

@app.route('/loaddataset/preprocess.html/', methods=['POST', 'GET'])
@app.route('/loaddataset/modelling.html/', methods=['POST', 'GET'])
def preprocess_back():
    return redirect('/')

@app.route('/preprocess.html/preprocess.html/')
@app.route('/preprocess.html/modelling.html/')
@app.route('/modelling.html/modelling.html/')
@app.route('/modelling.html/preprocess.html/')

def back():
    return redirect('/')

@app.route('/interpolation/', methods=['POST'])
def interpolate():
    global data
    if request.method== "POST":
        if request.form['interpolate']=='Interpolate':
            data = data.interpolate()
            data= data.replace(np.nan, 0)
            
            global message
            message= message + '\n Successfully fillled the null values with the mean/median/mode imputation depending on the data distribution \n'
            dtypes= data.dtypes
            columns= data.columns
            dataset_nulls= data.isnull().sum()
            return render_template('preprocess.html',dataset_nulls=dataset_nulls,columns=columns,dtypes=dtypes, message = message,data= data, dataset= data)
        return redirect('/interpolation/')
    return redirect('/interpolation/')


@app.route('/columnselection/', methods=['POST', 'GET'])
def ChooseColumn():
    global data
    try:
        if request.method== "POST":
    
            delete= request.form['column']
            data.drop([f'{delete}'], axis=1, inplace=True)
            #data = dataset
            dtypes= data.dtypes
            columns= data.columns
            dataset_nulls= data.isnull().sum()
            global message
            message= f'{message} \n The deleted field is {delete} \n'
            
        return render_template('preprocess.html', dataset_nulls=dataset_nulls,columns=columns,dtypes=dtypes, message = message, data= data, dataset= data)
    except Exception as e:
        message = e
    
    
@app.route('/to_numeric/', methods=['POST'])
def To_numeric():
    global data
    if request.method== "POST":

        to_numeric= request.form['to_numeric']
        dataset= data
       
        #global data
        data= pd.get_dummies(dataset, columns = [f'{to_numeric}'], drop_first= True)

        dtypes= data.dtypes
        columns= data.columns
        dataset_nulls= data.isnull().sum()
        global message
        message= f'{message} \n The converted field is {to_numeric} \n'
        
    return render_template('preprocess.html', dataset_nulls=dataset_nulls,columns=columns,dtypes=dtypes, message = message, data= data)



@app.route('/modelling.html/', methods=['POST', 'GET'])
def model():
    try:
        return render_template('modelling.html', data=data)
    except NameError:
        message= "Please load the dataset first!!"
        #return redirect(url_for('index'), message= message)
        return render_template("index.html", message = message)



@app.route('/optimalClusters/', methods=['POST'])
def NumClustersGrapgh():
    try:
        if request.method=='POST':
            if request.form['show_elbow']:
                
                wcss_list= []    
                for cluster in range(1, 11):  
                    kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state= 0)  
                    kmeans.fit(data)
                    wcss_list.append(kmeans.inertia_)
                    
                frame= pd.DataFrame({'Cluster': range(1,11), 'wcss_list': wcss_list})
                plt.figure(figsize=(10,10))
                plt.plot(frame['Cluster'], frame['wcss_list'], marker='o')
                plt.title('The Elobw Method Graph')  
                plt.xlabel('Number of clusters(k)')  
                plt.ylabel('wcss_list') 
                plt.savefig('static/images/elbow_graph/elbowGraph.png')
                plt.show()
                    
                    
                return redirect('/modelling.html/')
    except Exception as e:
        message= f"There are some errors or some features you selected are strings or objects. We advice to change them to be numeric at preprocessing stage"
        return render_template('modelling.html',message=message )
        
@app.route('/getElbowGraph/<elbow>/',methods=['GET'])
def elbow(elbow):
    return send_from_directory(app.config['IMG_DIRECTORY_ELBOWS'], elbow)



@app.route('/selectedModel/', methods=['POST', 'GET'])
def training_model():
    global data
    global label_counter
    try:
        if request.method=="POST":
            colors=['blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'skyblue', 'yellow', 'grey', 'orange']
            
            if request.form['algorithm']:
                algorithm = request.form['model']
                nclusters= request.form['nclusters']
                rstate= request.form['rstate']
                
                #label_counter= None
                #pred_clusters=None
                
                if algorithm == 'kmeans':
                    model= KMeans(n_clusters= int(nclusters), random_state= int(rstate), init= 'k-means++', algorithm='auto')
                    model= model.fit(data)
                    pred_clusters= model.predict(data)
                    cluster_centers= model.cluster_centers_
                    wcss= model.inertia_
                    n_iterations= model.n_iter_
                    label_counter = Counter(model.labels_)
                    message= 'successfully created some clusters'
                    plt.figure(figsize=(10,10))
                    
                    #sns.scatterplot(data=data, x="var1", y="var2", hue=model.labels_)
                    sns.scatterplot(data=data)
                    
                    
                        
                    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker="X", c="r", s=80, label="centroids")
                    plt.legend()
                    plt.savefig('static/images/segments/whole_dataset/scatterGraph.png')
                    plt.show()
                    
                    plt.bar(label_counter.keys(), label_counter.values(), color= colors )
                    plt.legend()
                    plt.xlabel('The Clusters')
                    plt.ylabel('Number of records')
                    plt.title('The clusters and their number of records for the whole dataset after preprocessing')
                    plt.savefig('static/images/segments/whole_dataset_bar/BarGraph.png')
                    plt.show()
                    
                elif algorithm == 'KNN':
                    message= 'KNN is mainly for Classification problems'
                elif algorithm == 'GaussianMixture':
                    gmm = GaussianMixture(n_components=int(nclusters))
                    gmm.fit(data)
                    message = 'gausian mixture'
                    
    
                else:
                    return render_template('modelling.html')
            return render_template('modelling.html',wcss=wcss, n_iterations = n_iterations, label_counter=label_counter, pred_clusters=pred_clusters,  message = message, algorithm=algorithm, nclusters=nclusters, rstate=rstate)
    except ValueError:
        message= "Sorry The Application can not cluster. There are some errors or some features you selected are strings or objects. We advice to change them to be numeric at preprocessing stage"
        return render_template('modelling.html',message=message )

@app.route('/selectedModel_Variables/', methods=['POST'])
def training_model_vars():
    global data
    if request.method=="POST":
        
        if request.form['cluster_vars']:
            algorithm = request.form['model']
            nclusters= request.form['nclusters']
            rstate= request.form['rstate']
            x_variable= request.form['x_var']
            y_variable= request.form['y_var']
            
           # X= data[[f'{x_variable}', f'{y_variable}']]
            X = data.loc[:, [f'{x_variable}', f'{y_variable}']].values
            
            
            #label_counter= None
            #pred_clusters=None
            
            if algorithm == 'kmeans':
                model= KMeans(n_clusters= int(nclusters), random_state= int(rstate), init= 'k-means++', algorithm='auto')
                model= model.fit(X)
                pred_clusters= model.predict(X)
                cluster_centers= model.cluster_centers_
                wcss= model.inertia_
                n_iterations= model.n_iter_
                
                label_counter = Counter(model.labels_)
                message= 'successfully created some clusters'
                
                colors=['blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'skyblue', 'yellow', 'grey', 'orange']

                plt.figure(figsize=(10,10))
                for i in range(int(nclusters)):
                    plt.scatter(X[pred_clusters==i, 0], X[pred_clusters==i, 1], s=40, c=colors[i], label= f'Cluster {i}')
                plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker="X", c="black", s=100, label="centroids")
                plt.title('SEGMENTS FOR CUSTOMERS plotting {} against {}'.format(y_variable, x_variable))
                plt.xlabel(f'{x_variable}')
                plt.ylabel(f'{y_variable}')
                plt.savefig('static/images/segments/variables/scatterGraph_Vars.png')
                plt.show()
                
                return render_template('modelling.html' ,wcss=wcss, n_iterations = n_iterations, y_variable=y_variable, x_variable = x_variable, label_counter=label_counter, pred_clusters=pred_clusters,  message = message, algorithm=algorithm, nclusters=nclusters, rstate=rstate)
            elif algorithm == 'GaussianMixture':
                gmm = GaussianMixture(n_components=int(nclusters))
                gmm.fit(data)
                
                #predictions from gmm
                labels = gmm.predict(data)
                #frame = pd.DataFrame(data)
                frame = data.loc[:, [f'{x_variable}', f'{y_variable}' ]].values
               # X = data.loc[:, [f'{x_variable}', f'{y_variable}']].values
                #frame['cluster'] = labels
                #frame.columns = [f'{x_variable}', f'{y_variable}', 'cluster']
                label_counter = Counter(labels)
                
                colors=['blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'skyblue', 'yellow', 'grey', 'orange']
                for k in range(int(nclusters)):
                    #frame = frame[frame["cluster"]==k]
                    plt.scatter(frame[labels==k, 0], frame[labels==k, 1], s=40, c=colors[k], label= f'Cluster {k}')
                plt.title('SEGMENTS FOR CUSTOMERS plotting {} against {}'.format(y_variable, x_variable))
                plt.xlabel(f'{x_variable}')
                plt.ylabel(f'{y_variable}')
                plt.savefig('static/images/segments/variables_gmm/scatterGraph_Vars.png')
                    
                plt.show()
                message =f'Successfully cluster {y_variable} against {x_variable} using Gaussian mixture'
                return render_template('modelling.html', y_variable=y_variable, x_variable = x_variable, label_counter=label_counter, pred_clusters=labels,  message = message, algorithm=algorithm, nclusters=nclusters, rstate=rstate)
            return redirect('modelling.html')
        return redirect('modelling.html')
                
@app.route('/visualize')
def visualize():
    return render_template('visualize.html')
    
@app.route('/visualizeGraph', methods=['POST', 'GET'])
def visualizegraph():
    if request.method=="POST":
        #if os.path.exists(app.config['IMG_DIRECTORY'] + 'scatterGraph.png'):
        filename= request.form['segments']
        graph= os.listdir(app.config['IMG_DIRECTORY'])
        BarGraph= os.listdir(app.config['IMG_DIRECTORY_BAR'])
        var_graph= os.listdir(app.config['IMG_DIRECTORY_VARS'])
        var_graph_gmm= os.listdir(app.config['IMG_DIRECTORY_VARS_GMM'])
        
        
        
        #graph= url_for('static', filename= 'images/' + filename)
        return render_template('visualize.html',BarGraph=BarGraph, graph= graph, var_graph= var_graph, var_graph_gmm=var_graph_gmm)
        #return redirect(url_for('static', filename= 'images/' + filename), code=301)

@app.route('/serve_image/<filename>', methods=['GET'])        
def serve_image(filename):
    try:
        return send_from_directory(app.config['IMG_DIRECTORY'], filename)
    except Exception as e:
        return redirect('/visualize')

@app.route('/servee_image/<filenamee>', methods=['GET'])        
def servee_image(filenamee):
    try:
        return send_from_directory(app.config['IMG_DIRECTORY_VARS'], filenamee)
    except Exception as e:
        return redirect('/visualize')
    
@app.route('/servee_image_gmm/<gmm>', methods=['GET'])        
def servee_image_gmm(gmm):
    try:
        return send_from_directory(app.config['IMG_DIRECTORY_VARS_GMM'], gmm)
    except Exception as e:
        return redirect('/visualize')
    
@app.route('/servee_image_bar/<bar>', methods=['GET'])        
def servee_imag_bar(bar):
    try:
        return send_from_directory(app.config['IMG_DIRECTORY_BAR'], bar)
    except Exception as e:
        return redirect('/visualize')
    
@app.route('/logout')
def logout():
    print('Logging out ....')
    
    if os.path.exists(app.config['IMG_DIRECTORY_ELBOWS'] + 'elbowGraph.png'):
        os.remove(app.config['IMG_DIRECTORY_ELBOWS'] + 'elbowGraph.png')
        os.system('cls')
        session['dataset']= 'Inactive'
        session.clear()
    else:
        message="The system can not find the paths you specified for the plot variables===delete"
        
    if os.path.exists(app.config['IMG_DIRECTORY_VARS'] + 'scatterGraph_Vars.png'):
        os.remove(app.config['IMG_DIRECTORY_VARS'] + 'scatterGraph_Vars.png')
        os.system('cls')
        session['dataset']= 'Inactive'
        session.clear()
        
        
    else:
        message="The system can not find the paths you specified for the plot elbow===delete"
        
    if os.path.exists(app.config['IMG_DIRECTORY'] + 'scatterGraph.png'):
        os.remove(app.config['IMG_DIRECTORY'] + 'scatterGraph.png')
        os.system('cls')
        session['dataset']= 'Inactive'
        session.clear()
        
        
    else:
        message="The system can not find the paths you specified for the plot wholedataset==delete"
        
    if os.path.exists(app.config['IMG_DIRECTORY_VARS_GMM'] + 'scatterGraph_Vars.png'):
        os.remove(app.config['IMG_DIRECTORY_VARS_GMM'] + 'scatterGraph_Vars.png')
        os.system('cls')
        session['dataset']= 'Inactive'
        session.clear()
        
        
    else:
        message="The system can not find the paths you specified for the plot wholedataset==delete"
        
    if os.path.exists(app.config['IMG_DIRECTORY_BAR'] + 'BarGraph.png'):
        os.remove(app.config['IMG_DIRECTORY_BAR'] + 'BarGraph.png')
        os.system('cls')
        session['dataset']= 'Inactive'
        session.clear()
        
        
    else:
        message="The system can not find the paths you specified for the plot wholedataset==delete"
        
        
    return redirect('/')

@app.route('/reading_documentation', methods= ['POST'])
def documentationn():
    if request.method=="POST":
        with open(app.config['DOCUMENTATION'] + 'documentation.pdf', 'rb') as file:
            document = file.read()
            
            return render_template('index.html',document= document)
        return redirect('/')
    return redirect('/')

@app.route('/stastics')
def statistics():
    try:
        global data
        global context
        context= {
            
            'columns': data.columns,
            'dtypes':data.dtypes,
            'dataset_nulls':data.isnull().sum()
        }
        return render_template('pure_statistics.html', context= context, collected = 'Nothing has been calculated yet')
    except Exception as e:
        return render_template('index.html', message='Please No dataset has been indicated', e= e)

@app.route('/calculte', methods=["POST"])
def calculate():
    global context
    if request.form['apply_stat']=='Calculate':
        
        ready= []
        
        for column in context['columns']:
            try:
                form_column= request.form[column]
                if column == form_column:
                    ready.append(column)
                else:
                    continue
            except Exception as e:
                continue
        result_total= []
        result_mean= []
        result_min= [] 
        result_max= []
        result_freq= []  
        if request.form['total']:
            for clmn in ready:
                if data[f'{clmn}'].dtype != object:
                    totals= data[f'{clmn}'].sum()
                    list_total= f' total for {clmn} is {totals}'
                    result_total.append(str(list_total))
        else:
            pass
        
        if request.form['mean']:
            for clmn in ready:
                if data[f'{clmn}'].dtype != object:
                    means= data[f'{clmn}'].mean()
                    list_means= f' mean for {clmn} is {means}'
                    result_mean.append(str(list_means))
        else:
            pass

        if request.form['min']:
            for clmn in ready:
                if data[f'{clmn}'].dtype != object:
                    mins= data[f'{clmn}'].min()
                    list_mins= f' minimum for {clmn} is {mins}'
                    result_min.append(str(list_mins))
        else:
            pass

        if request.form['max']:
            for clmn in ready:
                if data[f'{clmn}'].dtype != object :
                    maxs= data[f'{clmn}'].max()
                    list_max= f' minimum for {clmn} is {maxs}'
                    result_max.append(str(list_max))

        else:
            pass
        if request.form['quantity']:
            print('--------- quantity')
        else:
            pass
        
        if request.form['frequency']:
            for clmn in ready:
                if data[f'{clmn}'].dtype==object:
                    freq= data.groupby(f'{clmn}')[f'{clmn}'].count()
                    #df.groupby('colB')['colB'].count()
                    list_freq= f' minimum for {clmn} is {freq}'
                    result_freq.append(str(list_freq))
        else:
            pass
        
        if request.form['recency']:
            print('-----------')
        else:
            pass
        
        if request.form['top']:
            print('-----------')
        else:
            pass
        collected= {
            'result_total': result_total,
            'result_mean': result_mean,
            'result_min': result_min,
            'result_max': result_max,
            'result_freq': result_freq
        }    
        return render_template('pure_statistics.html', context= context, collected = collected )


        


    






if __name__ =='__main__':
    app.run(debug= False)
    
    
    
    
    
    
    
    
    
    