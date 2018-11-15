
import pandas as pd
import numpy as np
import os
import time
import h5py
import seaborn as sns
import matplotlib.pyplot as plt

from mouse_info import Mouse
import visual_behavior_research.utilities as dro
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator import foraging
from imaging_behavior.core.slicer import BinarySlicer

import imaging_behavior as ib
import imaging_behavior.core.utilities as ut
import imaging_behavior.io.manifest as manifest
import imaging_behavior.plotting.utilities as pu
import imaging_behavior.plotting.plotting_functions as pf


def event_triggered_lick_extraction(licktimes,time_array,events,time_before,time_after):
    '''sahar: copied from imaging_behavior.utilities, but changed to pass time before/after'''
    licks_by_trial = []
    for event in events:
        licks_by_trial.append(licktimes[np.logical_and(licktimes>=time_array[event]-int(time_before),
            licktimes<=time_array[event]+int(time_after))]-time_array[event])
    return licks_by_trial


class CreateWFSummaryFigure(object):
    def __init__(self, pixels, ProcessWFData_object=None, movie=None, session=None, df_jcam=None, manifest_dict=None, dff_movie=None, save_dir=None):

        if ProcessWFData_object is not None:
            movie = ProcessWFData_object.movie
            session = ProcessWFData_object.session
            df_jcam = ProcessWFData_object.dataframe
            manifest_dict = ProcessWFData_object.manifest_dict
            dff_movie = ProcessWFData_object.dff_movie

        self.movie = movie
        self.session = session
        self.df = df_jcam
        self.manifest_dict = manifest_dict
        self.dff_movie = dff_movie
        self.pixels = pixels
        self.colors = sns.color_palette()

        self.pkl_file = self.manifest_dict['behavioral_log_file_path']
        self.pkl_data = pd.read_pickle(self.pkl_file)
        self.get_event_triggered_avg_data()
        self.get_avg_movie_frames()
        



    def get_event_triggered_avg_data(self):
        time_array = np.hstack((0,np.cumsum(self.pkl_data['vsyncintervals'])))/1000.
        lickframes = ut.remove_consecutive(self.pkl_data['lickData'][0])
        licktimes = time_array[lickframes]

        data_l = []
        for row,trial_type in enumerate(['HIT','MISS','FA','CR']):
            change_df = self.df[(self.df.response_type == trial_type) & (self.df.trial_type != 'aborted')]
            change_events = (change_df.change_frame.values).astype(int)
            event_licks = event_triggered_lick_extraction(licktimes,time_array,change_events,time_before=0,time_after=1)
            event_licks = np.nanmean(ut.flatten_array(event_licks))
            events = change_df.stim_frame_jcam

            for pixel in self.pixels:
                mask = np.zeros((np.shape(self.movie)[1],np.shape(self.movie)[2]))*np.nan
                mask[pixel[1],pixel[0]] = 1            
                data = ut.event_triggered_average(self.movie,
                                                  mask,
                                                  events,
                                                  frames_before=150,
                                                  frames_after=100,
                                                  output='dff',
                                                  norm_frames=5,
                                                  progressbar=False)
                data = dict(trial_type=trial_type,
                            pixel=pixel,
                            trace_mean = data['trace_mean'],
                           event_licks_mean = event_licks,
                           all_traces = data['all_traces'])         
                datadata_l.append(data)
        data_l = pd.DataFrame(data_l)
        self.event_triggered_avg_data = data_l
        return self.event_triggered_avg_data

    def get_avg_movie_frames(self):
        change_df = self.df[(self.df.response_type == trial_type) & (self.df.trial_type != 'aborted')]
        vc_pixel = self.pixels[0]
        vc_data = self.event_triggered_avg_data[self.event_triggered_avg_data.pixel==vc_pixel]

        maxvals_l = []
        for row,trial_type in enumerate(['HIT','MISS','FA','CR']):
            events = change_df.stim_frame_jcam

            frames = ut.event_triggered_movie(self.movie,
                                          events,
                                          frames_before=200,
                                          frames_after=50,
                                          output='dff',
                                          norm_frames=200,
                                          progressbar=False)

            normalize_to = vc_data[vc_data.trial_type==trial_type]
            maxval = np.nanmax(np.abs(frames)) 
            max_norm = np.nanpercentile(np.abs(normalize_to['all_traces']),98)
            maxvals = dict(trial_type=trial_type,
                          maxval=maxval,
                          normalize_to=normalize_to,
                          max_norm=max_norm,
                          frames=frames)
            maxvals_l.append(maxvals)
        maxvals_l = pd.DataFrame(maxvals_l)  
        real_max = np.max(maxvals_l.max_norm)
        self.movie_maxvals = maxvals_l
        self.real_max_vals = real_max
        return self.movie_maxvals, self.real_max_vals



    def plot_brain_image(self, ax):
        ax.imshow(self.movie[5000,:,:],cmap='gray')
        ax.grid(False)
             
        for ii,pixel in enumerate(self.pixels):
            ax.plot(pixel[0],pixel[1],'o',color=self.colors[ii])
            ax.text(pixel[0]+self.movie.shape[2]/20,pixel[1],str(ii),color=self.colors[ii])
        ax.set_xlim(0,np.shape(self.movie)[2])
        ax.set_ylim(np.shape(self.movie)[1],0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])


#------------------------------------------------------------------------------------------------------------------------
def plot_event_timeseries(ax,frames_before=150,frames_after=100,progressbar=False,output='dff'):

    data_l = []
    for row,trial_type in enumerate(['HIT','MISS','FA','CR']):
        ax[row].set_title('Trial Type = {}'.format(trial_type))
        
        pkl_file = manifest_dict['behavioral_log_file_path']
        pkl_data = pd.read_pickle(pkl_file)
        time_array = np.hstack((0,np.cumsum(pkl_data['vsyncintervals'])))/1000.

        lickframes = ut.remove_consecutive(pkl_data['lickData'][0])
        licktimes = time_array[lickframes]

        change_events = (df[(df.response_type==trial_type)&(df.trial_type!='aborted')].change_frame.values).astype(int)
    
        event_licks = event_triggered_lick_extraction(licktimes,time_array,change_events,time_before=0,time_after=1)
        event_licks = np.nanmean(ut.flatten_array(event_licks))


        
        
        for pixel in pixels:
            mask = np.zeros((np.shape(movie)[1],np.shape(movie)[2]))*np.nan
            mask[pixel[1],pixel[0]]=1
            
            events = df[(df.response_type==trial_type)&(df.trial_type!='aborted')].stim_frame_jcam
            
            data = ut.event_triggered_average(movie,
                                              mask,
                                              events,
                                              frames_before,
                                              frames_after,
                                              output=output,
                                              norm_frames=5,
                                              progressbar=progressbar)
            
            ax[row].plot(data['t'],data['trace_mean'])
            if row == len(ax)-1:
                ax[row].set_xlabel('Time from stim (s)')
            ax[row].set_ylabel('$\Delta$F/F')
            
            data = dict(trial_type=trial_type,
                        pixel=pixel,
                        trace_mean = data['trace_mean'],
                       event_licks_mean = event_licks)
          
            data_l.append(data)
         
    data_l = pd.DataFrame(data_l)
    lick_matrix = data_l.pivot_table(values='event_licks_mean', columns='trial_type')
    
    max_yval = np.max(dro.flatten_list(list(data_l.trace_mean)))+.001
    min_yval = np.min(dro.flatten_list(list(data_l.trace_mean)))-.001
    min_xval = np.true_divide(frames_before,100)
    max_xval = np.true_divide(frames_after,100)
    
    for row,trial_type in enumerate(['HIT','MISS','FA','CR']):
        ax[row].axvline(0,color='k')
        for pixel in pixels:
            ax[row].set_ylim([min_yval,max_yval])
            ax[row].set_xlim([-min_xval,max_xval])
        
    ax[0].axvline(lick_matrix['HIT'][0],color='gray', linewidth=3, ls='dashed', alpha=.9)
    ax[2].axvline(lick_matrix['FA'][0],color='gray', linewidth=3, ls='dashed', alpha=.9)

    # ax[0].axvline(lick_matrix['HIT'])
    # ax[2].axvline(lick_matrix['FA'])



##sm_change: normalized all frames to max val of location of pixel 0 (in V1), baked in colorbars
def plot_movie_frames(fig,ax,movie,session,df, pixels,colors=sns.color_palette(),times=[0,0.1,0.2,0.3],
                          frames_before=200,frames_after=50,progressbar=False,output='dff',fs=100):
    
    maxvals_l = []
    
    t_array = np.arange(-1.*frames_before/fs,1.*frames_after/fs,1./fs)
    for row,trial_type in enumerate(['HIT','MISS','FA','CR']):
          
        events = df[(df.response_type==trial_type)&(df.trial_type!='aborted')].stim_frame_jcam

        frames = ut.event_triggered_movie(movie,
                                          events,
                                          frames_before,
                                          frames_after,
                                          output=output,
                                          norm_frames=200,
                                          progressbar=progressbar)
            
        pixel = pixels[0]
        mask = np.zeros((np.shape(movie)[1],np.shape(movie)[2]))*np.nan
        mask[pixel[1],pixel[0]]=1 
        
        normalize_to = ut.event_triggered_average(movie,
                                              mask,
                                              events,
                                              frames_before,
                                              frames_after,
                                              output=output,
                                              norm_frames=200,
                                              progressbar=progressbar)
        
        
        maxval = np.nanmax(np.abs(frames))
        
        max_norm = np.nanpercentile(np.abs(normalize_to['all_traces']),98)
        
        maxvals = dict(trial_type=trial_type,
                      maxval=maxval,
                      normalize_to=normalize_to,
                      max_norm=max_norm,
                      frames=frames)
        maxvals_l.append(maxvals)

    maxvals_l = pd.DataFrame(maxvals_l)  
    real_max = np.max(maxvals_l.max_norm)
    

    for row,trial_type in enumerate(['HIT','MISS','FA','CR']):
        for col,t in enumerate(times):
            frame = np.where(np.isclose(t_array,t))[0][0]
            ax[row][col].patch.set_visible(False)
            im=pf.show_image(maxvals_l.frames[row][frame,:,:],ax=ax[row][col],
                             cmin=-real_max, cmax=real_max,cmap='coolwarm', colorbar=False)
            if col == 3:
                im=pf.show_image(maxvals_l.frames[row][frame,:,:],ax=ax[row][col],
                             cmin=-real_max, cmax=real_max,cmap='coolwarm', colorbar=True)
            if row == 0:
                ax[row][col].set_title('Time From \nstim = {} s'.format(t))

##sm_change: made some aesthetic changes to the plots to make them easier to read, but nothing substantiative
def plot_full_timeseries(ax,movie,pixels,session,df,colors=sns.color_palette()):
    
    t=session.timeline.times['fluorescence_read_times']/60.
    t = t[:np.shape(movie)[0]]
    
    #=======================================================
    #get the times and cumulative volume
    times = df.starttime
    cumulative_volume = np.cumsum(df.number_of_rewards*df.reward_volume)

    #remove the nans
    trial_times_nonans = times[pd.isnull(cumulative_volume)==False]
    cumulative_volume_nonans = cumulative_volume[pd.isnull(cumulative_volume)==False]

    #get running speed
    ts,running_speed = ut.get_running_speed(session.behavior_log_dict)

    #resample cumulative volume on the running speed time series so they can share x-axes
    cv_dense = ut.resample(trial_times_nonans,cumulative_volume_nonans,ts)


    ax[0].plot(ts/60.,running_speed, '-k',)
    # ax[0].set_ylabel('running speed (cm/s)', color='k')
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('r')

    with sns.axes_style("dark"):
        ax2 = ax[0].twinx()
        ax2.plot(ts/60.,cv_dense,'-b',linewidth=3)
        # ax2.set_ylabel('Cumulative\nVolume (mL)',color='b')
        ax2.set_xlim(0,np.max(ts/60.))
        ax2.set_ylim(0,1.1*np.max(cumulative_volume_nonans))
        for tl in ax2.get_yticklabels():
            tl.set_color('b')
    
    #=======================================================
    df_nonaborted = df[df.trial_type!='aborted']
    hit_rate,fa_rate,d_prime = dro.get_response_rates(df_nonaborted,
                                                          sliding_window=100,
                                                          reward_window=session.behavior_log_dict['response_window'])

    ax[1].plot(df_nonaborted.starttime/60.,d_prime,color='black',linewidth=2)
    # ax[1].set_ylabel("d'")
    ax[1].set_ylim([-.5,4.5])
    
    #=======================================================
    y_vals = []
    for ii,pixel in enumerate(pixels):
        y_vals.append(movie[:,pixel[1],pixel[0]])
    y_vals = np.array(y_vals)
    scale = np.max(np.abs(y_vals))

    for ii,pixel in enumerate(pixels):
        ax[2].plot(t,ii*scale+y_vals[ii],color=colors[ii])
        
    ax[2].set_ylabel('pixel')
    ax[2].set_yticks(scale*np.arange(len(pixels)))
    ax[2].set_yticklabels(np.arange(len(pixels)))
    
    ax[2].plot([t[len(t)/10],t[len(t)/10]],[ii*scale-scale/2.,ii*scale+scale/2.],'-k',linewidth=3)
    ax[2].text(t[len(t)/11],ii*scale+scale/2.+scale/10,' $\Delta$F/F = {}'.format(round(scale,3)),va='top', weight='bold')
    
    #=======================================================
    for ii,title in enumerate(['Run Speed (cm/s); Cumulative Water (mL)',"Rolling d'",'Activity in defined locations']):
        ax[ii].set_title(title)
    
    ax[2].set_xlabel('Time (minutes)')

def plot_zoomed_timeseries(ax,movie,pixels,session,df,colors=sns.color_palette(),fs=100):
    
    t=session.timeline.times['fluorescence_read_times']/60.
    t = t[:np.shape(movie)[0]]
    
    y_vals = []
    for ii,pixel in enumerate(pixels):
        y_vals.append(movie[:,pixel[1],pixel[0]])
    y_vals = np.array(y_vals)
    scale = np.max(np.abs(y_vals))
    for ii,pixel in enumerate(pixels):
        ax.plot(t,ii*scale+y_vals[ii],color=colors[ii])
    
    first_idx = len(t)/10
    span = 60
    ax.set_xlim(t[first_idx],t[first_idx+span*fs])
    
    ax.plot([t[first_idx+span/5*fs],t[first_idx+span/5*fs]],[(ii+0)*scale-scale/2.,(ii+0)*scale+scale/2.],'-k',linewidth=3)
    ax.text(t[int(first_idx+span/4.5*fs)],(ii+0)*scale+scale/2.+scale/10,'$\Delta$F/F = {}'.format(round(scale,3)))
        
    ax.set_ylabel('pixel')
    ax.set_yticks(scale*np.arange(len(pixels)))
    ax.set_yticklabels(np.arange(len(pixels)))
    ax.set_xlabel('Time (minutes)')
    ax.set_title('zoomed in view on activity')


def make_table(ax,session,df, pixels):    
    # m_info = Mouse(session.mouse_id)
    # genotype = m_info.labtracks_info['pedigree_prefix']
    try:
        user_id = session.user_id
    except:
        user_id = 'unspecified'
    
    t0=time.time() 
    data = [['Mouse ID','session.mouse_id'],
            ['Genotype','genotype'],
            ['Date','pd.to_datetime(session.start_date_time).strftime("%m-%d-%Y")'],
            ['Time','pd.to_datetime(session.start_date_time).strftime("%H:%M")'],
            ['Task','session.behavior_log_dict["task"]'],
            ['Duration (min)','round(df.iloc[0]["session_duration"]/60.,2)'],
            ["Water Rec'd (ml)",'df["cumulative_volume"].max()'],
            ['Trained by','user_id'],
            ['Rig ID','str(df.iloc[0]["rig_id"])'],
            ['Pixel coords','pixels']]

    t0=time.time()
    cell_text = []
    for x in data:
        try:
            cell_text.append([eval(x[1])])
        except:
            cell_text.append([np.nan])

    #define row colors
    row_colors = [['lightgray'],['white']]*(len(data))

    t0=time.time()
     #make the table
    table = ax.table(cellText=cell_text,
                          rowLabels=[x[0] for x in data],
                          rowColours=dro.flatten_list(row_colors)[:len(data)],
                          colLabels=None,
                          loc='center',
                          cellLoc='left',
                          rowLoc='right',
                          cellColours=row_colors[:len(data)])

    t0=time.time()
    table.auto_set_font_size(False) 
    table.set_fontsize(9)
    table.scale(2,2)
    ax.grid(False)
    ax.patch.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    #do some cell resizing
    cell_dict=table.get_celld()
    for cell in cell_dict:
        cell_dict[cell].set_height(0.1)
        if cell[1] == -1:
            cell_dict[cell].set_width(0.2)
        if cell[1] == 0:
            cell_dict[cell].set_width(1.2)



def generate_summary_figure(movie, session, df_jcam, man, dff_movie, pixels, save_dir=None):

    
    print "initializing figure"
    t0=time.time()
    sns.set_style('darkgrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=(1.4*8.5,1.4*11))

    print "that took {} seconds".format(time.time()-t0)
    print ""

    print "setting up subfigures"
    t0=time.time()

    divider = 0.4
    wspace = 0.075
    ax_table = pu.placeAxesOnGrid(fig,xspan = (0,0.25),yspan=(0,0.2))
    ax_brain = pu.placeAxesOnGrid(fig,xspan = (0,0.25),yspan=(0.20,divider-wspace/2))
    ax_full_ts = pu.placeAxesOnGrid(fig,dim=(3,1),xspan = (0.35,0.7),yspan=(0.00,divider-wspace/2),sharex=True)
    ax_zoomed_ts = pu.placeAxesOnGrid(fig,dim=(1,1),xspan = (0.77,1),yspan=(0.00,divider-wspace/2),sharex=True)
    ax_event_ts = pu.placeAxesOnGrid(fig,dim=(4,1),xspan = (0,0.25),yspan=(divider+wspace/2,1),sharex=True,sharey=True)
    ax_movie_frames = pu.placeAxesOnGrid(fig,dim=(4,4),xspan = (0.3,0.95),yspan=(divider+wspace/2,1),sharex=True,sharey=True)

    print "that took {} seconds".format(time.time()-t0)
    print ""

    print "making table"
    t0=time.time()
    make_table(ax_table,session,df_jcam, pixels)
    print "that took {} seconds".format(time.time()-t0)
    print ""

    print "showing movie frame"
    t0=time.time()
    plot_brain_image(ax_brain,movie,pixels)
    print "that took {} seconds".format(time.time()-t0)
    print ""

    print "plotting full timeseries"
    t0=time.time()
    plot_full_timeseries(ax_full_ts,dff_movie,pixels,session,df_jcam)
    print "that took {} seconds".format(time.time()-t0)
    print ""

    print "plotting zoomed time series"
    t0=time.time()
    plot_zoomed_timeseries(ax_zoomed_ts,dff_movie,pixels,session,df_jcam)
    print "that took {} seconds".format(time.time()-t0)
    print ""

    print "plotting event timeseries"
    t0=time.time()
    plot_event_timeseries(ax_event_ts, movie, pixels, session, df_jcam, man)
    print "that took {} seconds".format(time.time()-t0)
    print ""

    print "displaying movie frames"
    t0=time.time()
    plot_movie_frames(fig,ax_movie_frames,movie,session,df_jcam,pixels=pixels)
    print "that took {} seconds".format(time.time()-t0)
    print ""

    if save_dir is not None:
        try:
            print "saving figure"
            t0=time.time()
            fig_name = '{}_{}_WF_summary_figure.png'.format(session.date.replace('-', '')[2:], session.mouse_id)
            fig.savefig(os.path.join(save_dir, fig_name), bbox_inches='tight',transparent=False)
            print "that took {} seconds".format(time.time()-t0)
            print ""
        except Exception as e:
            print "couldn't save figure because: {}".format(e)
    
    # summary_figure_savepath = ut.check_network_path_syntax("//aibsdata2/nc-ophys\CorticalMapping/IntrinsicImageData/WF_Summary_Plots")
    # fn = os.path.split(path)[-1]
    # shutil.copyfile(os.path.join(path,'summary_figure.png'),(os.path.join(summary_figure_savepath,fn+'_WF_summary_figure.png')))
