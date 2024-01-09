---
title: "Audio files"
layout: single
permalink: /audio_PAM
author_profile: false
header:
  overlay_color: "#000"
  overlay_filter: "0.6"
#   overlay_image: /images/particles.jpg
excerpt: "The recording was carried out in Aubervilliers with musicians from the 'Conservatoire à rayonnement régional'"
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---
<html>

<!-- <span style="color:red"><b>Warning: Due to authorization issues, we are not able to provide all the audio files yet. </b> </span> -->

</html>
# Perdrix

<!-- Here say something about mixing -->


<!-- Cette pièce a été pensée et écrite lors de son séjour londonien en 2020. On y retrouve des clins d’œil au climat britannique et cette atmosphère tout à fait particulière. Perdrix est divisée en plusieurs périodes qui retracent les étapes d’une journée. En effet, la direction de cette pièce est axée sur le lien entre le temps et la nature. On peut y entendre un instant de contemplation dédié à l’imagination et l’appréciation de la beauté de notre environnement. -->

*Perdrix* is an original composition from Inès Lassègue. This piece was imagined during her stay in London in 2020, and one can grasp some vibes of this british atmosphere. The piece is divided in several parts retracing the moments of the day. Indeed, the piece is based on the interaction between tile and nature. We can hear a moment of contemplation dedicated to the appreciation and the beauty of nature.

> Full mix

<html>
<audio controls>
  <source src="/audio/Perdrix.wav">
</audio>
</html>

# Source separation results

We tried our algorithm in different configurations to evaluate its efficiency. The algorithm takes a multichannel audio with all instrument playing as input, and produces 5 (1 per source) multichannel audios as if the instrument was playing alone in the room. Here we present the most interesting results for the different cases with all audios.

## Technical precision
The piece is written for a quintet: Violins, Flute, Clarinet and Cello. We used 8 microphones for the recording:
- 5 *spot* microphones close to each instrument (named after the instrument)
- 3 linked microphones to capture the global scene (named Left, Center, Right)

All the microphones are used for the separation.

## Separation on full mix without prior information

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <center>
    <strong> Original </strong>
    </center>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <center>
    <strong> Separated </strong>
    </center>
  </div>
</div>

</html>

> Violin 1

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/no_separation/micro_Violin1.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
  <source src="/audio/no_effect_audios/separation/micro_Violin1/source_3_micro_Violin1_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Violin 2

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/no_separation/micro_Violin2.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/separation/micro_Violin2/source_4_micro_Violin2_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Flute

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/no_separation/micro_Flute.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/separation/micro_Flute/source_0_micro_Flute_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Clarinet

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/no_separation/micro_Clarinet.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/separation/micro_Clarinet/source_1_micro_Clarinet_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Cello

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/no_separation/micro_Cello.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/no_effect_audios/separation/micro_Cello/source_2_micro_Cello_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>


## Separation on raw audio without prior information
<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <center>
    <strong> Original </strong>
    </center>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <center>
    <strong> Separated </strong>
    </center>
  </div>
</div>

</html>

> Violin 1

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/no_separation/micro_Violin1.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
  <source src="/audio/sto_audios/separation/micro_Violin1/source_3_micro_Violin1_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Violin 2

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/no_separation/micro_Violin2.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/separation/micro_Violin2/source_4_micro_Violin2_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Flute

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/no_separation/micro_Flute.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/separation/micro_Flute/source_0_micro_Flute_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Clarinet

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/no_separation/micro_Clarinet.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/separation/micro_Clarinet/source_1_micro_Clarinet_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Cello

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/no_separation/micro_Cello.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/sto_audios/separation/micro_Cello/source_2_micro_Cello_audio_length_12_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>


## Separation on raw audio with prior information
<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <center>
    <strong> Original </strong>
    </center>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <center>
    <strong> Separated </strong>
    </center>
  </div>
</div>

</html>

> Violin 1

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/no_separation/micro_Violin1.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
  <source src="/audio/prior_dictionnary/separation/micro_Violin1/source_3_micro_Violin1_audio_length_10_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Violin 2

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/no_separation/micro_Violin2.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/separation/micro_Violin2/source_4_micro_Violin2_audio_length_10_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Flute

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/no_separation/micro_Flute.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/separation/micro_Flute/source_0_micro_Flute_audio_length_10_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Clarinet

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/no_separation/micro_Clarinet.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/separation/micro_Clarinet/source_1_micro_Clarinet_audio_length_10_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>

> Cello

<html>

<div id="container">
  <div id="left-column">
    <!-- content for the left column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/no_separation/micro_Cello.wav">
    </audio>
  </div>
  <div id="right-column">
    <!-- content for the right column goes here -->
    <audio controls>
      <source src="/audio/prior_dictionnary/separation/micro_Cello/source_2_micro_Cello_audio_length_10_n_basis_32_n_fft_4096.wav">
    </audio>
  </div>
</div>

</html>
