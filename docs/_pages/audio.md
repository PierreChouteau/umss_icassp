---
title: "Audio Examples"
layout: single
permalink: /audio
author_profile: false
# classes: wide
header:
  overlay_color: "#000"
  overlay_filter: "0.6"
excerpt: "Experimental results" # "Here you can find the audio files of the different pieces we worked on." - Example of a subtitle
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---
<html>

</html>


Welcome to the audio examples page. Here you'll find the results of our experiments, as well as audio examples of music source separation.


<!-- For experiments where our architecture is compared to the [__US__](https://ieeexplore.ieee.org/document/10058592) model by Schulze-Foster _et al._ and the [__U-Net__](https://program.ismir2020.net/poster_5-14.html) model by Petermann _et al._, we show only our two best approaches:
- __VA_NN_2__
- __Warmup__ -->

All the audios presented below are taken from the evaluation datasets: __ChoralSingingDataset__ or __Cantoría__.


---
# Source separation results - Joint learning

## BC1song

For these results, we trained the different models on the __BC1Song__ dataset, and evaluated on the __ChoralSingingDataset__ dataset. These results are presented in the table 1a of the paper, and audio examples are shown below.

> Audio Mix Example
<audio controls>
  <source src="/audio/Apprentissage_Conjoint/Melange/mix_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
</audio>


> Separated voices from Wiener filtering
<html>
  <table>
    <thread>
      <tr>
        <th>
          <!-- <center> Voice </center> -->
        </th>
        <th>
          <center> Original </center>
        </th>
        <th>
          <center> UMSS </center>
        </th>
        <th>
          <center> U-Net </center>
        </th>
        <th>
          <center> S<sub>F</sub>S<sub>F</sub> </center>
        </th>
        <th>
          <center> S<sub>FT</sub>S<sub>FT</sub> </center>
        </th>
        <th>
          <center> S<sub>F</sub>S<sub>FT</sub> </center>
        </th>
        <th>
          <center> W<sub>UP</sub> </center>
        </th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <th> <strong> Soprano </strong> </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/Apprentissage_Conjoint/Original_sources/target_sources_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Ref_sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Unet_Sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_0/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_1/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/W_UP/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
      </tr>
     <tr>
        <th> <strong> Alto </strong> </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/Apprentissage_Conjoint/Original_sources/target_sources_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Ref_sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Unet_Sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_0/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_1/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/W_UP/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Tenor </strong> </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/Apprentissage_Conjoint/Original_sources/target_sources_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Ref_sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Unet_Sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_0/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_1/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/W_UP/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Bass </strong> </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/Apprentissage_Conjoint/Original_sources/target_sources_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Ref_sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Unet_Sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_0/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_1/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/W_UP/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Re-synthesized mix </strong> </th>
        <th></th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/Ref_sources/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
        <th></th>
         <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
         <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_0/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/VA_NN_1/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BC1song/W_UP/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
      </tr>
    </tbody>
  </table>
</html>
<br/>

## BCBSQ

For these results, we trained the different models on the __BCBSQ__ dataset, and evaluated on the __ChoralSingingDataset__ dataset. These results are presented in the table 1b of the paper, and audio examples are shown below.

> Audio Mix Example
<audio controls>
  <source src="/audio/Apprentissage_Conjoint/Melange/mix_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
</audio>


> Separated voices from Wiener filtering
<html>
  <table>
    <thread>
      <tr>
        <th>
          <!-- <center> Voice </center> -->
        </th>
        <th>
          <center> Original </center>
        </th>
        <th>
          <center> UMSS </center>
        </th>
        <th>
          <center> U-Net </center>
        </th>
        <th>
          <center> S<sub>F</sub>S<sub>F</sub> </center>
        </th>
        <th>
          <center> S<sub>FT</sub>S<sub>FT</sub> </center>
        </th>
        <th>
          <center> S<sub>F</sub>S<sub>FT</sub> </center>
        </th>
        <th>
          <center> W<sub>UP</sub> </center>
        </th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <th> <strong> Soprano </strong> </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/Apprentissage_Conjoint/Original_sources/target_sources_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Ref_sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Unet_Sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_0/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_1/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/W_UP/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_s.wav"/>
          </audio>
        </th>
      </tr>
     <tr>
        <th> <strong> Alto </strong> </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/Apprentissage_Conjoint/Original_sources/target_sources_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Ref_sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Unet_Sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_0/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_1/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/W_UP/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_a.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Tenor </strong> </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/Apprentissage_Conjoint/Original_sources/target_sources_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Ref_sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Unet_Sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_0/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_1/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/W_UP/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_t.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Bass </strong> </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/Apprentissage_Conjoint/Original_sources/target_sources_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Ref_sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Unet_Sources/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_0/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_1/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/W_UP/sources_estimates_masking_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984_voice_b.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Re-synthesized mix </strong> </th>
        <th></th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/Ref_sources/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
        <th></th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_0/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/VA_NN_1/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/Apprentissage_Conjoint/BCBQ/W_UP/mix_reconstruct_el_rossinyol_Sno_208_At_2-06_Tor2-09_Bos_207_27.984.wav"/>
          </audio>
        </th>
      </tr>
    </tbody>
  </table>
</html>
<br/><br/>


---
# Generalization capabilities on a new dataset: Cantoría

For these results, we trained the different models on the __BCBSQ__ dataset, and evaluated them on the __Cantoría__ dataset. These results are presented in the table 3 of the paper, and audio examples are shown below.


> Audio Mix Example
<audio controls="">
  <source src="/audio/cantoria/mix/mix_CEA_satb_15.984.wav"/>
</audio>


> Separated voices from Wiener filtering
<html>
  <table>
    <thread>
      <tr>
        <th>
          <!-- <center> Voice </center> -->
        </th>
         <th>
          <center> Original </center>
        </th>
        <th>
          <center> UMSS </center>
        </th>
        <th>
          <center> U-Net </center>
        </th>
        <th>
          <center> W<sub>UP</sub> </center>
        </th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <th> <strong> Soprano </strong> </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/target_sources/target_sources_CEA_satb_15.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio class="px-1" controls="" controlslist="nodownload">
            <source src="/audio/cantoria/BCBSQ/US/sources_estimates_masking_CEA_satb_15.984_voice_s.wav" type="audio/wav">
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/Unet/sources_estimates_masking_CEA_satb_15.984_voice_s.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/W-Up/sources_estimates_masking_CEA_satb_15.984_voice_s.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Alto </strong> </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/target_sources/target_sources_CEA_satb_15.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/US/sources_estimates_masking_CEA_satb_15.984_voice_a.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/Unet/sources_estimates_masking_CEA_satb_15.984_voice_a.wav"/>
          </audio>
        </th>
         <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/W-Up/sources_estimates_masking_CEA_satb_15.984_voice_a.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Tenor </strong> </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/target_sources/target_sources_CEA_satb_15.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/US/sources_estimates_masking_CEA_satb_15.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/Unet/sources_estimates_masking_CEA_satb_15.984_voice_t.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/W-Up/sources_estimates_masking_CEA_satb_15.984_voice_t.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Bass </strong> </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/target_sources/target_sources_CEA_satb_15.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/US/sources_estimates_masking_CEA_satb_15.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/Unet/sources_estimates_masking_CEA_satb_15.984_voice_b.wav"/>
          </audio>
        </th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/W-Up/sources_estimates_masking_CEA_satb_15.984_voice_b.wav"/>
          </audio>
        </th>
      </tr>
      <tr>
        <th> <strong> Re-synthesized mix </strong> </th>
        <th></th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/US/mix_reconstruct_CEA_satb_15.984.wav"/>
          </audio>
        </th>
        <th></th>
        <th>
          <audio controls="">
            <source src="/audio/cantoria/BCBSQ/W-Up/mix_reconstruct_CEA_satb_15.984.wav"/>
          </audio>
        </th>
      </tr>
    </tbody>
  </table>
</html>