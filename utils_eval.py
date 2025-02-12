import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_data(path):
    df = pd.read_pickle(path)
    return df


def load_data_per_voice(path):
    df = pd.read_pickle(path)
    
    df_soprano = df[df['voice'] == 's']
    df_alto = df[df['voice'] == 'a']
    df_tenor = df[df['voice'] == 't']
    df_bass = df[df['voice'] == 'b']
    
    return df_soprano, df_alto, df_tenor, df_bass


def get_stats(df):
    means = df.mean(axis=0, skipna=True, numeric_only=True)
    medians = df.median(axis=0, skipna=True, numeric_only=True)
    stds = df.std(axis=0, skipna=True, numeric_only=True)
    
    return means, medians, stds


def print_stats(means, medians, stds, gen_synth= False, masking_synth=True, voicing=False, f0=True):
    
    if gen_synth:
        print('sp_SNR [mean, median, std]:', means['sp_SNR'], medians['sp_SNR'], stds['sp_SNR'])
        print('sp_SI-SNR [mean, median, std]:', means['sp_SI-SNR'], medians['sp_SI-SNR'], stds['sp_SI-SNR'])
        print('mel_cep_dist [mean, median, std]:', means['mel_cep_dist'], medians['mel_cep_dist'], stds['mel_cep_dist'])
    
    if masking_synth:
        print('SI-SDR [mean, median, std]:', means['SI-SDR'], medians['SI-SDR'], stds['SI-SDR'])
        print('sp_SNR [mean, median, std]:', means['sp_SNR'], medians['sp_SNR'], stds['sp_SNR'])
        print('sp_SI-SNR [mean, median, std]:', means['sp_SI-SNR'], medians['sp_SI-SNR'], stds['sp_SI-SNR'])
        print('mel_cep_dist [mean, median, std]:', means['mel_cep_dist'], medians['mel_cep_dist'], stds['mel_cep_dist'])
        
    if voicing:
        print('Voicing Recall:', 'mean', means['Voicing-Recall'], 'median', medians['Voicing-Recall'], 'std', stds['Voicing-Recall'])
        print('Voicing False Alarm:', 'mean', means['Voicing-False-Alarm'], 'median', medians['Voicing-False-Alarm'], 'std', stds['Voicing-False-Alarm'])
    
    if f0:
        print('f0 Raw Pitch Accuracy:', 'mean', means['Raw-Pitch-Accuracy'], 'median', medians['Raw-Pitch-Accuracy'], 'std', stds['Raw-Pitch-Accuracy'])
        print('f0 Overall Accuracy:', 'mean', means['Overall-Accuracy'], 'median', medians['Overall-Accuracy'], 'std', stds['Overall-Accuracy'])
        print('f0 Raw Chroma Accuracy:', 'mean', means['Raw-Chroma-Accuracy'], 'median', medians['Raw-Chroma-Accuracy'], 'std', stds['Raw-Chroma-Accuracy'])
        print('f0 F-Score:', 'mean', means['F-Score'], 'median', medians['F-Score'], 'std', stds['F-Score'])



def sns_plot_f0_metric_per_model(df, dataset_name, metric, legend=True, save_fig=False):
    
    sns.set_theme(context='notebook', style='whitegrid')

    g = sns.boxplot(x='Modèle', y=metric, hue='voice', data=df, palette=sns.color_palette("flare", 4), showfliers=False)

    # Change axis labels
    g.set(xlabel="", ylabel=metric)

    if legend:
        # Change legend position
        sns.move_legend(
            g, "lower center",
            bbox_to_anchor=(.5, 1), 
            ncol=4,
            title=None, 
            frameon=False,
        )

        # Change legend labels
        g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        for t, l in zip(g.legend_.texts, ['Soprano', 'Alto', 'Tenor', 'Bass']): t.set_text(l)

    # set figure size in inches
    plt.gcf().set_size_inches(w=6.718, h=3)

    # tight layout
    plt.tight_layout()
    
    # save figure
    if save_fig: plt.savefig('./resultats_figs/{}_{}.pdf'.format(dataset_name, metric), dpi=600)
    plt.show()
    
    
def sns_plot_separation_metric_per_model(df, dataset_name, metric, w=6.718, h=2.5, save_fig=False, save_path='./resultats_figs/'): 
    sns.set_theme(context='notebook', style='whitegrid')

    g = sns.barplot(x='Modèle', y=metric, hue='voice', data=df, palette=sns.color_palette("flare", 4), errorbar=('ci',95), errwidth=1, capsize=0.05)

    # Change axis labels
    g.set(xlabel="", ylabel=metric +" [dB]")

    # Change legend position
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), 
        ncol=4,
        title=None, 
        frameon=False,
    )

    # Change legend labels
    g.legend(loc='lower center', bbox_to_anchor=(.5, 1), ncol=4)
    for t, l in zip(g.legend_.texts, ['Soprano', 'Alto', 'Tenor', 'Bass']): t.set_text(l)

    # set figure size in inches
    plt.gcf().set_size_inches(w=w, h=h)

    # tight layout
    plt.tight_layout()
    
    # save figure
    if save_fig: plt.savefig(save_path+'{}_{}.pdf'.format(dataset_name, metric), dpi=600)
    plt.show()
    


def sns_plot_separation_metric_per_song(df, fig_name, metric, w=6.718, h=2.5, save_fig=False, save_path='./resultats_figs/'):
    
    sns.set_theme(context='notebook', style='whitegrid')

    g = sns.barplot(x='mix_name', y=metric, hue='voice', data=df, palette=sns.color_palette("flare", 4), errorbar=('ci',95), errwidth=1, capsize=0.05)

    # Change axis labels
    g.set(xlabel="", ylabel=metric +" [dB]")

    # Change legend position
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), 
        ncol=4,
        title=None, 
        frameon=False,
    )

    # Change legend labels
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    for t, l in zip(g.legend_.texts, ['Soprano', 'Alto', 'Tenor', 'Bass']): t.set_text(l)

    # set figure size in inches
    plt.gcf().set_size_inches(w=w, h=h)

    # tight layout
    plt.tight_layout()

    # save figure
    if save_fig: plt.savefig(save_path+'{}_{}.pdf'.format(fig_name, metric), dpi=600)
    plt.show()
    
    
def sns_plot_f0_metric_per_song(df, fig_name, metric, w=6.718, h=2.5, legend=True, save_fig=False, save_path='./resultats_figs/'):
    
    sns.set_theme(context='notebook', style='whitegrid')

    g = sns.boxplot(x='mix_name', y=metric, hue='voice', data=df, palette=sns.color_palette("flare", 4), showfliers=False)

    # Change axis labels
    g.set(xlabel="", ylabel=metric +" [dB]")

    if legend:
        # Change legend position
        sns.move_legend(
            g, "lower center",
            bbox_to_anchor=(.5, 1), 
            ncol=4,
            title=None, 
            frameon=False,
        )

        # Change legend labels
        g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        for t, l in zip(g.legend_.texts, ['Soprano', 'Alto', 'Tenor', 'Bass']): t.set_text(l)

    # set figure size in inches
    plt.gcf().set_size_inches(w=w, h=h)

    # tight layout
    plt.tight_layout()

    # save figure
    if save_fig: plt.savefig(save_path+'{}_{}.pdf'.format(fig_name, metric), dpi=600)
    plt.show()