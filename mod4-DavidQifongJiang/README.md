# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


Sentiment Classification:
Epoch 1, loss 31.464710860350277, train accuracy: 46.22%
Validation accuracy: 57.00%
Best Valid accuracy: 57.00%
Epoch 2, loss 31.15589339801301, train accuracy: 50.22%
Validation accuracy: 50.00%
Best Valid accuracy: 57.00%
Epoch 3, loss 30.982848965335243, train accuracy: 53.56%
Validation accuracy: 51.00%
Best Valid accuracy: 57.00%
Epoch 4, loss 30.82147436385636, train accuracy: 57.56%
Validation accuracy: 57.00%
Best Valid accuracy: 57.00%
Epoch 5, loss 30.642580909395505, train accuracy: 57.33%
Validation accuracy: 58.00%
Best Valid accuracy: 58.00%
Epoch 6, loss 30.477088118473343, train accuracy: 56.44%
Validation accuracy: 50.00%
Best Valid accuracy: 58.00%
Epoch 7, loss 30.230988976185078, train accuracy: 59.33%
Validation accuracy: 56.00%
Best Valid accuracy: 58.00%
Epoch 8, loss 29.958743074644357, train accuracy: 62.22%
Validation accuracy: 51.00%
Best Valid accuracy: 58.00%
Epoch 9, loss 29.531268846705046, train accuracy: 64.00%
Validation accuracy: 56.00%
Best Valid accuracy: 58.00%
Epoch 10, loss 29.254592439200483, train accuracy: 65.56%
Validation accuracy: 58.00%
Best Valid accuracy: 58.00%
Epoch 11, loss 28.768796393702353, train accuracy: 66.44%
Validation accuracy: 67.00%
Best Valid accuracy: 67.00%
Epoch 12, loss 28.302098594123827, train accuracy: 67.33%
Validation accuracy: 59.00%
Best Valid accuracy: 67.00%
Epoch 13, loss 27.94682004746461, train accuracy: 70.00%
Validation accuracy: 65.00%
Best Valid accuracy: 67.00%
Epoch 14, loss 27.216617765733154, train accuracy: 72.22%
Validation accuracy: 62.00%
Best Valid accuracy: 67.00%
Epoch 15, loss 26.82149299881866, train accuracy: 72.67%
Validation accuracy: 66.00%
Best Valid accuracy: 67.00%
Epoch 16, loss 26.014210069040242, train accuracy: 73.56%
Validation accuracy: 66.00%
Best Valid accuracy: 67.00%
Epoch 17, loss 25.33034936338562, train accuracy: 70.89%
Validation accuracy: 65.00%
Best Valid accuracy: 67.00%
Epoch 18, loss 24.778655041827342, train accuracy: 74.89%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 19, loss 24.492411509412456, train accuracy: 76.44%
Validation accuracy: 59.00%
Best Valid accuracy: 68.00%
Epoch 20, loss 23.422844774255577, train accuracy: 77.56%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 21, loss 22.40753470440263, train accuracy: 77.78%
Validation accuracy: 62.00%
Best Valid accuracy: 68.00%
Epoch 22, loss 21.862645077005673, train accuracy: 78.44%
Validation accuracy: 64.00%
Best Valid accuracy: 68.00%
Epoch 23, loss 20.85720473860382, train accuracy: 80.67%
Validation accuracy: 65.00%
Best Valid accuracy: 68.00%
Epoch 24, loss 20.55975208099362, train accuracy: 80.00%
Validation accuracy: 64.00%
Best Valid accuracy: 68.00%
Epoch 25, loss 20.400073566466105, train accuracy: 76.67%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 26, loss 20.194369171973545, train accuracy: 78.67%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 27, loss 19.208226401917358, train accuracy: 80.00%
Validation accuracy: 66.00%
Best Valid accuracy: 68.00%
Epoch 28, loss 19.195030228475012, train accuracy: 78.89%
Validation accuracy: 66.00%
Best Valid accuracy: 68.00%
Epoch 29, loss 18.696250053887425, train accuracy: 78.89%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 30, loss 17.564751147971254, train accuracy: 82.89%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%
Epoch 31, loss 17.03094483743679, train accuracy: 82.00%
Validation accuracy: 64.00%
Best Valid accuracy: 70.00%
Epoch 32, loss 17.02470688219658, train accuracy: 83.56%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 33, loss 16.663863054048907, train accuracy: 84.00%
Validation accuracy: 62.00%
Best Valid accuracy: 71.00%
Epoch 34, loss 15.608060384788311, train accuracy: 84.22%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 35, loss 15.621335920820048, train accuracy: 84.67%
Validation accuracy: 65.00%
Best Valid accuracy: 72.00%
Epoch 36, loss 15.300648792290492, train accuracy: 84.89%
Validation accuracy: 69.00%
Best Valid accuracy: 72.00%
Epoch 37, loss 15.08265147847234, train accuracy: 84.00%
Validation accuracy: 70.00%
Best Valid accuracy: 72.00%
Epoch 38, loss 14.861150745274415, train accuracy: 84.44%
Validation accuracy: 69.00%
Best Valid accuracy: 72.00%
Epoch 39, loss 14.944419505798503, train accuracy: 84.89%
Validation accuracy: 63.00%
Best Valid accuracy: 72.00%
Epoch 40, loss 14.404634048186669, train accuracy: 84.89%
Validation accuracy: 71.00%
Best Valid accuracy: 72.00%
Epoch 41, loss 13.815752234655362, train accuracy: 85.78%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 42, loss 13.416073931496081, train accuracy: 85.56%
Validation accuracy: 67.00%
Best Valid accuracy: 73.00%
Epoch 43, loss 13.649729141303576, train accuracy: 85.11%
Validation accuracy: 65.00%
Best Valid accuracy: 73.00%
Epoch 44, loss 12.835279711646802, train accuracy: 86.67%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 45, loss 13.328628801911373, train accuracy: 86.67%
Validation accuracy: 67.00%
Best Valid accuracy: 73.00%
Epoch 46, loss 13.945105265258096, train accuracy: 86.44%
Validation accuracy: 66.00%
Best Valid accuracy: 73.00%
Epoch 47, loss 13.237900322814403, train accuracy: 83.78%
Validation accuracy: 66.00%
Best Valid accuracy: 73.00%
Epoch 48, loss 12.60712840673885, train accuracy: 86.67%
Validation accuracy: 77.00%
Best Valid accuracy: 77.00%
Epoch 49, loss 11.901540261774269, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 77.00%
Epoch 50, loss 11.194220330970378, train accuracy: 87.33%
Validation accuracy: 62.00%
Best Valid accuracy: 77.00%
Epoch 51, loss 11.360687472090996, train accuracy: 87.56%
Validation accuracy: 70.00%
Best Valid accuracy: 77.00%
Epoch 52, loss 11.269507160123926, train accuracy: 87.56%
Validation accuracy: 68.00%
Best Valid accuracy: 77.00%
Epoch 53, loss 10.411387755376907, train accuracy: 89.56%
Validation accuracy: 74.00%
Best Valid accuracy: 77.00%

Mnist:
Epoch 10 loss 0.8723352131765988 valid acc 16/16
Epoch 10 loss 0.7539313357672338 valid acc 15/16
Epoch 10 loss 0.8401355621466805 valid acc 16/16
Epoch 10 loss 1.3349518845632988 valid acc 14/16
Epoch 10 loss 0.9672377721915515 valid acc 14/16
Epoch 10 loss 1.2444788251217198 valid acc 15/16
Epoch 10 loss 1.4488693735987637 valid acc 14/16
Epoch 10 loss 0.36140843320054894 valid acc 14/16
Epoch 10 loss 0.9176637897257006 valid acc 14/16
Epoch 10 loss 0.6751775304627723 valid acc 15/16
Epoch 10 loss 1.1647957028964702 valid acc 15/16
Epoch 10 loss 0.9372330448726298 valid acc 14/16
Epoch 10 loss 1.075585119254681 valid acc 14/16
Epoch 10 loss 1.809630611378695 valid acc 15/16
Epoch 10 loss 0.7610525315079817 valid acc 13/16
Epoch 10 loss 0.969297412049811 valid acc 15/16
Epoch 10 loss 1.8445800787446531 valid acc 16/16
Epoch 10 loss 0.5640721338156074 valid acc 16/16
Epoch 10 loss 0.9085761000442749 valid acc 15/16
Epoch 10 loss 1.3364653405942566 valid acc 15/16
Epoch 10 loss 1.0194839690571325 valid acc 14/16
Epoch 10 loss 0.870577618297069 valid acc 15/16
Epoch 10 loss 0.862412991182002 valid acc 14/16
Epoch 10 loss 0.8180066573807646 valid acc 15/16
Epoch 10 loss 0.8990390585674755 valid acc 14/16
Epoch 10 loss 0.7558016096826895 valid acc 15/16
Epoch 10 loss 1.2597235045827697 valid acc 14/16
Epoch 10 loss 1.0431877706818353 valid acc 14/16
Epoch 10 loss 0.7987850111978914 valid acc 15/16
Epoch 10 loss 0.6406783426565545 valid acc 15/16
Epoch 10 loss 1.0472234769903381 valid acc 14/16
Epoch 10 loss 0.8619759064060136 valid acc 14/16
Epoch 10 loss 0.9812388721017431 valid acc 14/16
Epoch 10 loss 0.7120479429161906 valid acc 14/16
Epoch 10 loss 0.9912446986383199 valid acc 14/16
Epoch 11 loss 0.18380862366720002 valid acc 14/16
Epoch 11 loss 1.2344813905030723 valid acc 15/16
Epoch 11 loss 1.0031233927716745 valid acc 15/16
Epoch 11 loss 0.5718097854247892 valid acc 15/16
Epoch 11 loss 0.6619354102751209 valid acc 14/16
Epoch 11 loss 0.9356553896889989 valid acc 15/16
Epoch 11 loss 1.0621049840171548 valid acc 15/16
Epoch 11 loss 1.661559314146835 valid acc 14/16
Epoch 11 loss 1.1618089821846032 valid acc 14/16
Epoch 11 loss 0.8365890399588363 valid acc 15/16
Epoch 11 loss 0.8923727006974009 valid acc 15/16
Epoch 11 loss 1.379345570824344 valid acc 14/16
Epoch 11 loss 0.8480948199170202 valid acc 14/16
Epoch 11 loss 1.2679660988270367 valid acc 14/16
Epoch 11 loss 1.1373740402579722 valid acc 14/16
Epoch 11 loss 0.6171535849703969 valid acc 14/16
Epoch 11 loss 1.561348842270303 valid acc 15/16
Epoch 11 loss 1.7961397626106614 valid acc 12/16
Epoch 11 loss 1.694638751406914 valid acc 15/16
Epoch 11 loss 1.6146163368531254 valid acc 15/16
Epoch 11 loss 0.942932882827908 valid acc 15/16
Epoch 11 loss 0.6474924520152763 valid acc 14/16
Epoch 11 loss 0.4015705556259498 valid acc 14/16
Epoch 11 loss 1.0581228090592893 valid acc 15/16
Epoch 11 loss 0.3637200426980828 valid acc 16/16
Epoch 11 loss 0.8794746644509708 valid acc 13/16
Epoch 11 loss 0.7997486185801348 valid acc 15/16
Epoch 11 loss 1.1920122425532256 valid acc 16/16
Epoch 11 loss 0.6607560953904028 valid acc 16/16
Epoch 11 loss 0.24532454606178572 valid acc 15/16
Epoch 11 loss 0.8091974992984079 valid acc 15/16
Epoch 11 loss 1.0010333151635848 valid acc 14/16
Epoch 11 loss 0.8019839090090637 valid acc 14/16
Epoch 11 loss 0.23583457215099246 valid acc 14/16
Epoch 11 loss 1.654173198086533 valid acc 15/16
Epoch 11 loss 0.6680153480060776 valid acc 16/16
Epoch 11 loss 0.406396689122198 valid acc 15/16
Epoch 11 loss 0.77195928787572 valid acc 15/16
Epoch 11 loss 1.202842763286578 valid acc 16/16
Epoch 11 loss 0.7836486401770832 valid acc 15/16
Epoch 11 loss 0.6486075542670533 valid acc 15/16
Epoch 11 loss 1.0988942665768686 valid acc 16/16
Epoch 11 loss 0.491217815199943 valid acc 16/16
Epoch 11 loss 0.346555513386388 valid acc 15/16
Epoch 11 loss 1.0872926159714715 valid acc 15/16
Epoch 11 loss 0.3366652430874826 valid acc 16/16
Epoch 11 loss 1.051999465073584 valid acc 15/16
Epoch 11 loss 1.0336966584794307 valid acc 16/16
Epoch 11 loss 0.6174415852922625 valid acc 15/16
Epoch 11 loss 0.9304303145031989 valid acc 14/16
Epoch 11 loss 0.7098970607638508 valid acc 15/16
Epoch 11 loss 0.94283753858729 valid acc 14/16
Epoch 11 loss 0.7558723415746267 valid acc 15/16
Epoch 11 loss 0.6963012606322257 valid acc 14/16
Epoch 11 loss 1.1807983378409792 valid acc 13/16
Epoch 11 loss 0.6591237119042234 valid acc 16/16
Epoch 11 loss 1.1268803849067235 valid acc 16/16
Epoch 11 loss 0.8478872978774623 valid acc 15/16
Epoch 11 loss 1.0496095286910119 valid acc 15/16
Epoch 11 loss 0.8611228202580485 valid acc 15/16
Epoch 11 loss 0.5307559798625383 valid acc 16/16
Epoch 11 loss 0.7661910472248037 valid acc 15/16
Epoch 11 loss 1.414500191202333 valid acc 14/16
Epoch 12 loss 0.08427519235178418 valid acc 16/16
Epoch 12 loss 1.0155571048645007 valid acc 14/16
Epoch 12 loss 1.2740666317073734 valid acc 14/16
Epoch 12 loss 0.6502475993264548 valid acc 14/16
Epoch 12 loss 0.5288285434260749 valid acc 14/16
Epoch 12 loss 0.7790330736969134 valid acc 13/16
Epoch 12 loss 1.4380297379890572 valid acc 12/16
Epoch 12 loss 1.2941421946139138 valid acc 13/16
Epoch 12 loss 1.6511557522081226 valid acc 15/16
Epoch 12 loss 0.639976731111753 valid acc 15/16
Epoch 12 loss 1.0834387211059038 valid acc 15/16
Epoch 12 loss 1.6281388059771595 valid acc 15/16
Epoch 12 loss 1.2996087630882744 valid acc 16/16
Epoch 12 loss 0.8981757830418169 valid acc 16/16
Epoch 12 loss 0.7917295870566697 valid acc 16/16
Epoch 12 loss 0.32460511073091436 valid acc 16/16
Epoch 12 loss 1.7694450296555575 valid acc 14/16
Epoch 12 loss 1.746040842119912 valid acc 14/16
Epoch 12 loss 1.3002964853791033 valid acc 15/16
Epoch 12 loss 1.4634893189659008 valid acc 15/16
Epoch 12 loss 0.8956786821092539 valid acc 15/16
Epoch 12 loss 0.7164008492384895 valid acc 14/16
Epoch 12 loss 0.3288534856921594 valid acc 14/16
Epoch 12 loss 0.9027789872578804 valid acc 12/16
Epoch 12 loss 0.6353234649090479 valid acc 14/16
Epoch 12 loss 1.3065737441017387 valid acc 16/16
Epoch 12 loss 0.8526127064084399 valid acc 14/16
Epoch 12 loss 1.0482563749433755 valid acc 14/16
Epoch 12 loss 0.9389885051096366 valid acc 14/16
Epoch 12 loss 0.5920218957218267 valid acc 15/16
Epoch 12 loss 0.8601528970441498 valid acc 14/16
Epoch 12 loss 1.3515477484195815 valid acc 14/16
Epoch 12 loss 0.5605434345132507 valid acc 14/16
Epoch 12 loss 0.8472440676747811 valid acc 14/16
Epoch 12 loss 1.3901060528016775 valid acc 16/16
Epoch 12 loss 1.2498514402193477 valid acc 15/16
Epoch 12 loss 1.1648307750518614 valid acc 15/16
Epoch 12 loss 0.8907898281475897 valid acc 15/16
Epoch 12 loss 1.2553230295865707 valid acc 16/16
Epoch 12 loss 1.0900620040627347 valid acc 15/16
Epoch 12 loss 0.6457439265186734 valid acc 15/16
Epoch 12 loss 1.541993149865143 valid acc 14/16
Epoch 12 loss 0.7074254852177755 valid acc 15/16
Epoch 12 loss 1.2578819480160377 valid acc 15/16
Epoch 12 loss 1.6175749584966397 valid acc 15/16
Epoch 12 loss 0.7166027382520959 valid acc 16/16
Epoch 12 loss 0.9906862150636695 valid acc 14/16
Epoch 12 loss 1.6661455875490552 valid acc 16/16
Epoch 12 loss 0.6228748784967693 valid acc 15/16
Epoch 12 loss 0.5864524441080454 valid acc 15/16
Epoch 12 loss 1.062428311168188 valid acc 15/16
Epoch 12 loss 0.781899281172108 valid acc 14/16
Epoch 12 loss 1.1344155897852408 valid acc 15/16
Epoch 12 loss 0.8821916574296754 valid acc 16/16
Epoch 12 loss 1.665378068173879 valid acc 15/16
Epoch 12 loss 0.7661063770902239 valid acc 14/16
Epoch 12 loss 1.4860277555701171 valid acc 15/16
Epoch 12 loss 0.8385191896940803 valid acc 14/16
Epoch 12 loss 1.414589884559211 valid acc 14/16
Epoch 12 loss 0.8559483703242957 valid acc 13/16
Epoch 12 loss 0.8727418896651928 valid acc 15/16
Epoch 12 loss 0.8123971496158497 valid acc 14/16
Epoch 12 loss 1.424793465835613 valid acc 14/16
Epoch 13 loss 0.16024091033277754 valid acc 14/16
Epoch 13 loss 0.5874932182688185 valid acc 15/16
Epoch 13 loss 0.9510642305669037 valid acc 15/16
Epoch 13 loss 0.653491605219615 valid acc 14/16
Epoch 13 loss 0.6095565749024995 valid acc 13/16
Epoch 13 loss 1.6758254827891892 valid acc 14/16
Epoch 13 loss 1.2193451963098827 valid acc 13/16
Epoch 13 loss 1.4805331545984746 valid acc 15/16
Epoch 13 loss 0.9679745386816181 valid acc 14/16
Epoch 13 loss 0.5292455746988887 valid acc 15/16
Epoch 13 loss 0.948453152636993 valid acc 14/16
Epoch 13 loss 1.7736115106196244 valid acc 14/16
Epoch 13 loss 0.8855198061701808 valid acc 13/16
Epoch 13 loss 0.8169216233291241 valid acc 13/16
Epoch 13 loss 0.940223037877971 valid acc 16/16
Epoch 13 loss 0.8492079774466831 valid acc 15/16
Epoch 13 loss 2.1228531007818807 valid acc 14/16
Epoch 13 loss 1.7805707855007191 valid acc 15/16
Epoch 13 loss 2.0190387360976074 valid acc 16/16
Epoch 13 loss 1.7307869232731548 valid acc 14/16
Epoch 13 loss 0.6106122978549844 valid acc 13/16
Epoch 13 loss 0.929756144339963 valid acc 14/16
Epoch 13 loss 0.6106248019921001 valid acc 16/16
Epoch 13 loss 1.0142068987742667 valid acc 13/16
Epoch 13 loss 0.9281338381717477 valid acc 15/16
Epoch 13 loss 2.1287901346710587 valid acc 15/16
Epoch 13 loss 1.3283378313732113 valid acc 15/16
Epoch 13 loss 0.8781451849101114 valid acc 14/16
Epoch 13 loss 1.0669540549755632 valid acc 14/16
Epoch 13 loss 0.5239407309095132 valid acc 15/16
Epoch 13 loss 0.6321587105731017 valid acc 15/16
Epoch 13 loss 2.194527969078421 valid acc 13/16
Epoch 13 loss 1.2351385627989628 valid acc 14/16
Epoch 13 loss 1.4701014571069728 valid acc 13/16
Epoch 13 loss 1.4939901986483244 valid acc 16/16
Epoch 13 loss 1.1060047031071016 valid acc 15/16
Epoch 13 loss 1.0148587726754166 valid acc 16/16
Epoch 13 loss 0.5845324184338292 valid acc 16/16
Epoch 13 loss 0.9424467464306987 valid acc 15/16
Epoch 13 loss 0.8403863400030595 valid acc 15/16
Epoch 13 loss 0.5858037431229823 valid acc 15/16
Epoch 13 loss 1.214013679012294 valid acc 14/16
Epoch 13 loss 0.6634202698612016 valid acc 14/16
Epoch 13 loss 0.8601215513212639 valid acc 15/16
Epoch 13 loss 1.2233478519856564 valid acc 16/16
Epoch 13 loss 0.6098761446662144 valid acc 15/16
Epoch 13 loss 0.7825844819307777 valid acc 15/16
Epoch 13 loss 0.9460884768700866 valid acc 15/16
Epoch 13 loss 0.9079392829255603 valid acc 14/16
Epoch 13 loss 0.8424855890488253 valid acc 15/16
Epoch 13 loss 0.8463901198239374 valid acc 15/16
Epoch 13 loss 1.0273208873846906 valid acc 12/16
Epoch 13 loss 0.8576598037693541 valid acc 12/16
Epoch 13 loss 0.7849777582512896 valid acc 13/16
Epoch 13 loss 0.9160996531884259 valid acc 14/16
Epoch 13 loss 0.5512595830522838 valid acc 12/16
Epoch 13 loss 1.3375794052421925 valid acc 15/16
Epoch 13 loss 1.0135054797222338 valid acc 15/16
Epoch 13 loss 1.0464333865626356 valid acc 14/16
Epoch 13 loss 1.208773603884301 valid acc 14/16
Epoch 13 loss 0.6979700836877403 valid acc 13/16
Epoch 13 loss 0.9476368682990526 valid acc 14/16
Epoch 13 loss 0.8877432991443857 valid acc 15/16
Epoch 14 loss 0.03511212326017238 valid acc 15/16
Epoch 14 loss 0.7888614575467575 valid acc 14/16
Epoch 14 loss 1.500563470726339 valid acc 14/16
Epoch 14 loss 0.39760044673604805 valid acc 15/16
Epoch 14 loss 0.5053185509342548 valid acc 12/16
Epoch 14 loss 1.1718149104816777 valid acc 12/16
Epoch 14 loss 0.9586371213093206 valid acc 13/16
Epoch 14 loss 1.2059458537499488 valid acc 14/16
Epoch 14 loss 1.1618414036902676 valid acc 14/16
Epoch 14 loss 0.6871068201941117 valid acc 15/16
Epoch 14 loss 0.9418975246302307 valid acc 13/16
Epoch 14 loss 1.320745538939147 valid acc 13/16
Epoch 14 loss 1.1032061810593856 valid acc 13/16
Epoch 14 loss 1.2285577808511476 valid acc 12/16
Epoch 14 loss 1.3919751788426074 valid acc 15/16
Epoch 14 loss 0.5372778614609471 valid acc 14/16
Epoch 14 loss 1.8006209201562948 valid acc 14/16
Epoch 14 loss 1.185078004389086 valid acc 14/16
Epoch 14 loss 1.734999528106357 valid acc 16/16
Epoch 14 loss 0.9111578411603869 valid acc 15/16
Epoch 14 loss 1.035003267958862 valid acc 14/16
Epoch 14 loss 0.7705411188345652 valid acc 14/16
Epoch 14 loss 0.3982721613344202 valid acc 15/16
Epoch 14 loss 0.8700624099511404 valid acc 15/16
Epoch 14 loss 0.8223079325904243 valid acc 15/16
Epoch 14 loss 1.0999963866608504 valid acc 16/16
Epoch 14 loss 0.8771418559798569 valid acc 14/16
Epoch 14 loss 0.857712212465853 valid acc 16/16
Epoch 14 loss 0.9178845607928267 valid acc 15/16
Epoch 14 loss 0.37501169015906954 valid acc 16/16
Epoch 14 loss 0.5122609227925039 valid acc 16/16
Epoch 14 loss 0.624230956697754 valid acc 16/16
Epoch 14 loss 1.622313178503887 valid acc 14/16
Epoch 14 loss 2.0547709768129603 valid acc 15/16
Epoch 14 loss 1.7978854291953388 valid acc 14/16
Epoch 14 loss 0.9190266360222226 valid acc 14/16
Epoch 14 loss 0.31968733408012034 valid acc 16/16
Epoch 14 loss 0.5992700068985883 valid acc 15/16
Epoch 14 loss 1.0331476218855602 valid acc 16/16
Epoch 14 loss 0.5885897044856363 valid acc 16/16
Epoch 14 loss 1.1096577110579293 valid acc 16/16
Epoch 14 loss 1.2331545924269023 valid acc 15/16
Epoch 14 loss 0.6757444512863743 valid acc 15/16
Epoch 14 loss 0.726829960776444 valid acc 16/16
Epoch 14 loss 1.3870312553997788 valid acc 15/16
Epoch 14 loss 0.3175274643428214 valid acc 16/16
Epoch 14 loss 0.974634975164503 valid acc 16/16
Epoch 14 loss 0.9148607665316684 valid acc 16/16
Epoch 14 loss 0.5256523534549564 valid acc 15/16
Epoch 14 loss 0.7598692367117488 valid acc 15/16
Epoch 14 loss 0.7033733811340911 valid acc 16/16
Epoch 14 loss 1.1587046951676923 valid acc 15/16
Epoch 14 loss 0.7597271599700537 valid acc 16/16
Epoch 14 loss 1.1549326047340949 valid acc 15/16
Epoch 14 loss 0.6709262376611859 valid acc 15/16
Epoch 14 loss 0.9424902019394472 valid acc 15/16
Epoch 14 loss 1.21027241956119 valid acc 16/16
Epoch 14 loss 1.0534904917610233 valid acc 15/16
Epoch 14 loss 0.9688363974128614 valid acc 14/16
Epoch 14 loss 0.773797697571839 valid acc 15/16
Epoch 14 loss 1.1687875243434707 valid acc 15/16
Epoch 14 loss 1.0952129247877107 valid acc 15/16
Epoch 14 loss 1.0321261574299743 valid acc 15/16
Epoch 15 loss 0.08252814067141018 valid acc 14/16
Epoch 15 loss 0.9128477352731447 valid acc 16/16
Epoch 15 loss 1.3695911257092683 valid acc 16/16
Epoch 15 loss 0.6050260089783175 valid acc 16/16
Epoch 15 loss 1.7018530852451212 valid acc 16/16
Epoch 15 loss 1.0933312561832336 valid acc 16/16
Epoch 15 loss 0.8311608073504282 valid acc 16/16
Epoch 15 loss 1.152274146884071 valid acc 13/16
Epoch 15 loss 1.1150537881033895 valid acc 16/16
Epoch 15 loss 0.6834200154521682 valid acc 15/16
Epoch 15 loss 0.988097922370268 valid acc 15/16
Epoch 15 loss 1.6755664745794574 valid acc 16/16
Epoch 15 loss 1.5998615723220435 valid acc 15/16
Epoch 15 loss 1.0930274500512216 valid acc 14/16
Epoch 15 loss 1.4491171274821109 valid acc 15/16
Epoch 15 loss 0.4895456929205775 valid acc 16/16
Epoch 15 loss 1.5295839255820414 valid acc 16/16
Epoch 15 loss 1.340601642677153 valid acc 15/16
Epoch 15 loss 1.406196769468595 valid acc 14/16
Epoch 15 loss 1.618000237737075 valid acc 15/16
Epoch 15 loss 0.9572393099940559 valid acc 15/16
Epoch 15 loss 0.7710707282387019 valid acc 13/16
Epoch 15 loss 0.3720754069358039 valid acc 13/16
Epoch 15 loss 0.5516666913737491 valid acc 15/16
Epoch 15 loss 0.9412305199439485 valid acc 15/16
Epoch 15 loss 1.2421582657375074 valid acc 15/16
Epoch 15 loss 1.4030126165765835 valid acc 16/16
Epoch 15 loss 1.1313685982407484 valid acc 16/16
Epoch 15 loss 1.2358274881574116 valid acc 15/16
Epoch 15 loss 1.2974173639984874 valid acc 15/16
Epoch 15 loss 1.2403401641527698 valid acc 13/16
Epoch 15 loss 1.1425021062413894 valid acc 15/16
Epoch 15 loss 1.1546153403410482 valid acc 15/16
Epoch 15 loss 1.5573402954428244 valid acc 15/16
Epoch 15 loss 2.0644843921594114 valid acc 16/16
Epoch 15 loss 0.8408733160326004 valid acc 15/16
Epoch 15 loss 0.7544859752823341 valid acc 13/16
Epoch 15 loss 0.9580790887415586 valid acc 14/16
Epoch 15 loss 1.1761525694228332 valid acc 15/16
Epoch 15 loss 0.751887520607001 valid acc 15/16
Epoch 15 loss 0.7813747932760232 valid acc 14/16
Epoch 15 loss 1.1063203858033819 valid acc 13/16
Epoch 15 loss 0.4284508072840363 valid acc 14/16
Epoch 15 loss 0.8723323509760486 valid acc 14/16
Epoch 15 loss 1.4318284221439657 valid acc 15/16
Epoch 15 loss 0.5648671725762512 valid acc 16/16
Epoch 15 loss 0.8852397090538651 valid acc 16/16
Epoch 15 loss 1.2675646211600642 valid acc 15/16
Epoch 15 loss 0.6598465982945149 valid acc 14/16
Epoch 15 loss 0.7037285291078248 valid acc 14/16
Epoch 15 loss 0.8237387588042206 valid acc 14/16
Epoch 15 loss 1.5142966176253516 valid acc 15/16
Epoch 15 loss 0.7840108113097839 valid acc 15/16
Epoch 15 loss 0.6649780874696771 valid acc 14/16
Epoch 15 loss 1.1837759941900796 valid acc 14/16
Epoch 15 loss 0.5919537184587564 valid acc 13/16
Epoch 15 loss 1.0663326193964533 valid acc 15/16
Epoch 15 loss 0.7997929171438902 valid acc 14/16
Epoch 15 loss 0.8922962152434504 valid acc 13/16
Epoch 15 loss 1.3361633736704566 valid acc 14/16
Epoch 15 loss 1.3085045394031927 valid acc 12/16
Epoch 15 loss 0.9255341707906157 valid acc 14/16
Epoch 15 loss 1.3788550547515814 valid acc 14/16
Epoch 16 loss 0.1666298815722852 valid acc 15/16
Epoch 16 loss 0.9109458796387362 valid acc 14/16
Epoch 16 loss 1.075035939891317 valid acc 13/16
Epoch 16 loss 0.8218823924155676 valid acc 13/16
Epoch 16 loss 1.1328525566918581 valid acc 14/16
Epoch 16 loss 1.0936147762085309 valid acc 11/16
Epoch 16 loss 1.0868866444580425 valid acc 14/16
Epoch 16 loss 1.262850914834165 valid acc 14/16
Epoch 16 loss 0.9195309709098247 valid acc 13/16
Epoch 16 loss 0.8474051192945221 valid acc 14/16
Epoch 16 loss 1.0596133555579068 valid acc 13/16
Epoch 16 loss 1.0212005066496574 valid acc 15/16
Epoch 16 loss 1.1377423092052834 valid acc 14/16
Epoch 16 loss 1.2176946775523887 valid acc 11/16
Epoch 16 loss 1.254805344854726 valid acc 14/16
Epoch 16 loss 0.43388183984761863 valid acc 15/16
Epoch 16 loss 1.3040816509551776 valid acc 15/16
Epoch 16 loss 1.0898919784741317 valid acc 14/16
Epoch 16 loss 2.6687442184181203 valid acc 14/16
Epoch 16 loss 1.691744874041667 valid acc 15/16
Epoch 16 loss 1.1832010142219624 valid acc 12/16
Epoch 16 loss 0.7968921456553972 valid acc 15/16
Epoch 16 loss 0.4425536266949759 valid acc 15/16
Epoch 16 loss 0.6545635983364242 valid acc 15/16
Epoch 16 loss 0.9158685731754657 valid acc 14/16
Epoch 16 loss 1.2515839348318458 valid acc 14/16
Epoch 16 loss 1.0177949931611674 valid acc 14/16
Epoch 16 loss 0.6272721370459742 valid acc 14/16
Epoch 16 loss 0.7011251581200333 valid acc 15/16
Epoch 16 loss 0.4975153754942906 valid acc 15/16
Epoch 16 loss 0.8397078774397528 valid acc 15/16
Epoch 16 loss 1.3936696735349159 valid acc 15/16
Epoch 16 loss 0.9887647449314021 valid acc 15/16
Epoch 16 loss 0.8709755797934092 valid acc 14/16
Epoch 16 loss 1.7230213119084326 valid acc 16/16
Epoch 16 loss 0.5888073297906748 valid acc 14/16
Epoch 16 loss 0.6249900011638152 valid acc 14/16
Epoch 16 loss 0.7325320643098945 valid acc 15/16
Epoch 16 loss 0.5000441890727092 valid acc 15/16
Epoch 16 loss 1.1164129208404312 valid acc 16/16
Epoch 16 loss 0.6770583675491431 valid acc 15/16
Epoch 16 loss 1.143871288308732 valid acc 15/16
Epoch 16 loss 0.42778720953993626 valid acc 15/16
Epoch 16 loss 0.3564586117473936 valid acc 15/16
Epoch 16 loss 1.2278152060821887 valid acc 14/16
Epoch 16 loss 0.19875412060635966 valid acc 15/16
Epoch 16 loss 1.6184113328688485 valid acc 15/16
Epoch 16 loss 0.6654806598929187 valid acc 15/16
Epoch 16 loss 0.8087959531426141 valid acc 15/16
Epoch 16 loss 0.6204764295935519 valid acc 15/16
Epoch 16 loss 1.0817767831504979 valid acc 15/16
Epoch 16 loss 0.9961312074994864 valid acc 14/16
Epoch 16 loss 1.1507391805392517 valid acc 14/16
Epoch 16 loss 0.6685870724346772 valid acc 13/16
Epoch 16 loss 0.931838653379262 valid acc 16/16
Epoch 16 loss 0.5725720265381917 valid acc 16/16
Epoch 16 loss 1.4833948060559516 valid acc 15/16
Epoch 16 loss 0.6501691768660791 valid acc 14/16
Epoch 16 loss 0.6336982033403278 valid acc 14/16
Epoch 16 loss 0.9138085191387606 valid acc 14/16
Epoch 16 loss 0.574501173810956 valid acc 13/16
Epoch 16 loss 1.0515302671504423 valid acc 15/16
Epoch 16 loss 0.8986662531286801 valid acc 15/16
Epoch 17 loss 0.015377053784008221 valid acc 15/16
Epoch 17 loss 1.1968353742545546 valid acc 16/16
Epoch 17 loss 1.5005894823025263 valid acc 14/16
Epoch 17 loss 0.5875446160379969 valid acc 15/16
Epoch 17 loss 0.4770637156106823 valid acc 15/16
Epoch 17 loss 1.7914509387499598 valid acc 13/16
Epoch 17 loss 1.0615008294275583 valid acc 15/16
Epoch 17 loss 1.4351008620849726 valid acc 16/16
Epoch 17 loss 1.2695132873448163 valid acc 15/16
Epoch 17 loss 0.47381931557591106 valid acc 14/16
Epoch 17 loss 0.8963268541859868 valid acc 14/16
Epoch 17 loss 1.0681149326847563 valid acc 15/16
Epoch 17 loss 1.3062129715423783 valid acc 13/16
Epoch 17 loss 1.1620056269494987 valid acc 14/16
Epoch 17 loss 0.9115259779924982 valid acc 14/16
Epoch 17 loss 0.6461010992017666 valid acc 13/16
Epoch 17 loss 1.5663623145448073 valid acc 15/16
Epoch 17 loss 1.2244071491496231 valid acc 14/16
Epoch 17 loss 1.9761931933660608 valid acc 15/16
Epoch 17 loss 1.707799446425486 valid acc 14/16
Epoch 17 loss 0.9314985105035551 valid acc 12/16
Epoch 17 loss 1.0243447509088242 valid acc 15/16
Epoch 17 loss 0.6051758399495499 valid acc 14/16
Epoch 17 loss 0.7408129667973209 valid acc 15/16
Epoch 17 loss 0.8802026198768877 valid acc 14/16
Epoch 17 loss 0.8227128617326267 valid acc 16/16
Epoch 17 loss 1.2255820438700795 valid acc 16/16
Epoch 17 loss 0.6702756987041754 valid acc 15/16
Epoch 17 loss 1.334408654451281 valid acc 14/16
Epoch 17 loss 0.53153433002746 valid acc 14/16
Epoch 17 loss 0.49755621125140864 valid acc 15/16
Epoch 17 loss 1.3625075703717022 valid acc 14/16
Epoch 17 loss 0.531147796706573 valid acc 14/16
Epoch 17 loss 1.1989330832971117 valid acc 14/16
Epoch 17 loss 2.218082078707572 valid acc 12/16
Epoch 17 loss 0.7282722560920637 valid acc 15/16
Epoch 17 loss 0.6995273817096014 valid acc 14/16
Epoch 17 loss 0.9579545284219366 valid acc 16/16
Epoch 17 loss 1.8094841134497461 valid acc 15/16
Epoch 17 loss 1.2025154765428097 valid acc 15/16
Epoch 17 loss 0.7214445013151256 valid acc 14/16
Epoch 17 loss 1.075346204613451 valid acc 15/16
Epoch 17 loss 0.37687774258101786 valid acc 15/16
Epoch 17 loss 0.8070941850129225 valid acc 16/16
Epoch 17 loss 2.3220626883003295 valid acc 14/16
Epoch 17 loss 0.23549521218112035 valid acc 14/16
Epoch 17 loss 1.4493512950515817 valid acc 14/16
Epoch 17 loss 1.9143919343288136 valid acc 14/16
Epoch 17 loss 1.3937278933521051 valid acc 14/16
Epoch 17 loss 0.7415882154054478 valid acc 13/16
Epoch 17 loss 0.8735722051021031 valid acc 15/16
Epoch 17 loss 1.0326947089302738 valid acc 16/16
Epoch 17 loss 0.8692808862761255 valid acc 14/16
Epoch 17 loss 0.7436960200238683 valid acc 16/16
Epoch 17 loss 1.0857995384183239 valid acc 16/16
Epoch 17 loss 1.1450986544193171 valid acc 14/16
Epoch 17 loss 1.2023709192146688 valid acc 13/16
Epoch 17 loss 0.8249378380167522 valid acc 14/16
Epoch 17 loss 0.8480969896231236 valid acc 15/16
Epoch 17 loss 1.0980075525622235 valid acc 15/16
Epoch 17 loss 0.5724907710998097 valid acc 15/16
Epoch 17 loss 1.2053539390436807 valid acc 15/16
Epoch 17 loss 1.3663330360217298 valid acc 13/16
Epoch 18 loss 0.08867569555322574 valid acc 14/16
Epoch 18 loss 1.1773025123682923 valid acc 13/16
Epoch 18 loss 1.2318909626011534 valid acc 14/16
Epoch 18 loss 0.7882870884865874 valid acc 15/16
Epoch 18 loss 0.5035991343150685 valid acc 13/16
Epoch 18 loss 1.1866809285613475 valid acc 14/16
Epoch 18 loss 0.8223991709524416 valid acc 15/16
Epoch 18 loss 1.9576090878248835 valid acc 15/16
Epoch 18 loss 1.107543804857466 valid acc 14/16
Epoch 18 loss 0.8642979684243897 valid acc 15/16
Epoch 18 loss 0.9013040593986434 valid acc 15/16
Epoch 18 loss 1.424926489111089 valid acc 16/16
Epoch 18 loss 0.6291913716521046 valid acc 16/16
Epoch 18 loss 1.6054560804651892 valid acc 14/16
Epoch 18 loss 0.9606410523507549 valid acc 14/16
Epoch 18 loss 0.9308675562023404 valid acc 15/16
Epoch 18 loss 0.9529614113691429 valid acc 14/16
Epoch 18 loss 0.909909667973964 valid acc 16/16
Epoch 18 loss 1.9300736584197078 valid acc 16/16
Epoch 18 loss 1.2011548500856009 valid acc 16/16
Epoch 18 loss 1.0302920148674826 valid acc 15/16
Epoch 18 loss 0.5619372618319919 valid acc 16/16
Epoch 18 loss 0.2982430310903878 valid acc 15/16
Epoch 18 loss 0.8157684617327958 valid acc 15/16
Epoch 18 loss 1.0346837983445252 valid acc 13/16
Epoch 18 loss 0.9478709708543329 valid acc 13/16
Epoch 18 loss 0.9906317342041003 valid acc 14/16
Epoch 18 loss 1.8337146245154763 valid acc 13/16
Epoch 18 loss 3.029871107571594 valid acc 15/16
Epoch 18 loss 0.347167358469207 valid acc 15/16
Epoch 18 loss 1.5404646768896746 valid acc 14/16
Epoch 18 loss 3.6665303908640707 valid acc 11/16
Epoch 18 loss 1.6559950339743554 valid acc 15/16
Epoch 18 loss 1.2326378252456403 valid acc 15/16
Epoch 18 loss 2.0437748935014177 valid acc 14/16
Epoch 18 loss 0.5876577744158196 valid acc 15/16
Epoch 18 loss 0.952089901588788 valid acc 15/16
Epoch 18 loss 0.7841984538659805 valid acc 15/16
Epoch 18 loss 1.3011646603652194 valid acc 15/16
Epoch 18 loss 1.049752889691976 valid acc 16/16
Epoch 18 loss 0.7553608089800629 valid acc 15/16
Epoch 18 loss 1.7291860716510843 valid acc 15/16
Epoch 18 loss 0.8470510225587232 valid acc 15/16
Epoch 18 loss 1.3341994305359652 valid acc 16/16
Epoch 18 loss 1.7826942007639408 valid acc 16/16
Epoch 18 loss 0.46812000122271236 valid acc 15/16
Epoch 18 loss 1.1176924670036494 valid acc 15/16
Epoch 18 loss 0.9028641113392897 valid acc 16/16
Epoch 18 loss 0.392272830991707 valid acc 16/16
Epoch 18 loss 0.7349929164650556 valid acc 16/16
Epoch 18 loss 1.0883613709697777 valid acc 14/16
Epoch 18 loss 1.591319797740158 valid acc 15/16
Epoch 18 loss 0.9597092992474083 valid acc 15/16
Epoch 18 loss 0.6905022350170826 valid acc 15/16
Epoch 18 loss 0.9862754597718248 valid acc 14/16
Epoch 18 loss 1.2398043406779133 valid acc 11/16
Epoch 18 loss 1.0436397471029262 valid acc 14/16
Epoch 18 loss 1.2822175817893844 valid acc 14/16
Epoch 18 loss 0.9478400189381841 valid acc 16/16
Epoch 18 loss 1.173209991541836 valid acc 13/16
Epoch 18 loss 0.9809981131226992 valid acc 15/16
Epoch 18 loss 0.7168703202995824 valid acc 16/16
Epoch 18 loss 1.3885318928061632 valid acc 15/16
Epoch 19 loss 0.05940374093229428 valid acc 15/16
Epoch 19 loss 0.7843549679501698 valid acc 15/16
Epoch 19 loss 1.1399858733030281 valid acc 16/16
Epoch 19 loss 1.4019979133198404 valid acc 13/16
Epoch 19 loss 0.7250693575555327 valid acc 13/16
Epoch 19 loss 1.1292605393826232 valid acc 15/16
Epoch 19 loss 0.7495351325127956 valid acc 14/16
Epoch 19 loss 1.4342008593210132 valid acc 15/16
Epoch 19 loss 2.4326312578892906 valid acc 13/16
Epoch 19 loss 0.5065174430102265 valid acc 13/16
Epoch 19 loss 1.1267339458940457 valid acc 14/16
Epoch 19 loss 1.5272052641182396 valid acc 16/16
Epoch 19 loss 0.7884735975993608 valid acc 15/16
Epoch 19 loss 1.1290982733508237 valid acc 16/16
Epoch 19 loss 0.8223805662938544 valid acc 16/16
Epoch 19 loss 0.7365671703224339 valid acc 16/16
Epoch 19 loss 1.2360005751480785 valid acc 16/16
Epoch 19 loss 1.4732966631531632 valid acc 16/16
Epoch 19 loss 0.9945943446062016 valid acc 16/16
Epoch 19 loss 0.809336507697763 valid acc 16/16
Epoch 19 loss 1.2305221617861417 valid acc 15/16
Epoch 19 loss 0.9122211884735699 valid acc 16/16
Epoch 19 loss 0.4081277084858263 valid acc 14/16
Epoch 19 loss 1.1403839769853108 valid acc 14/16
Epoch 19 loss 1.0410344737066426 valid acc 14/16
Epoch 19 loss 1.152931044908533 valid acc 12/16
Epoch 19 loss 0.6924289975553664 valid acc 15/16
Epoch 19 loss 1.0009880050220652 valid acc 15/16
Epoch 19 loss 1.399741416872686 valid acc 15/16
Epoch 19 loss 0.3803216671243657 valid acc 15/16
Epoch 19 loss 1.0828657799592356 valid acc 14/16
Epoch 19 loss 1.2917267555811323 valid acc 15/16
Epoch 19 loss 1.7350876175748842 valid acc 14/16
Epoch 19 loss 1.2422142659371427 valid acc 16/16
Epoch 19 loss 2.0794886360034077 valid acc 16/16
Epoch 19 loss 0.5475047704346874 valid acc 14/16
Epoch 19 loss 0.8046085509249264 valid acc 16/16
Epoch 19 loss 0.8512582131160319 valid acc 16/16
Epoch 19 loss 1.1599698445580429 valid acc 15/16
Epoch 19 loss 0.5612245617303918 valid acc 15/16
Epoch 19 loss 1.3483282000309473 valid acc 16/16
Epoch 19 loss 1.9218268647686236 valid acc 16/16
Epoch 19 loss 1.0949055486178967 valid acc 15/16
Epoch 19 loss 1.0499188447937755 valid acc 16/16
Epoch 19 loss 1.7723402074759351 valid acc 14/16
Epoch 19 loss 0.7012871238313259 valid acc 15/16
Epoch 19 loss 1.116085962188493 valid acc 16/16
Epoch 19 loss 0.8636797846017492 valid acc 14/16
Epoch 19 loss 1.3108045508062456 valid acc 15/16
Epoch 19 loss 1.2211238005076537 valid acc 15/16
Epoch 19 loss 1.075990392265211 valid acc 15/16
Epoch 19 loss 1.1891762404601776 valid acc 15/16
Epoch 19 loss 0.9282065604818684 valid acc 15/16
Epoch 19 loss 0.5132525108963202 valid acc 15/16
Epoch 19 loss 1.996142007823711 valid acc 16/16
Epoch 19 loss 0.7867454853467607 valid acc 16/16
Epoch 19 loss 0.9443985349131885 valid acc 16/16
Epoch 19 loss 1.0899455904044029 valid acc 16/16
Epoch 19 loss 0.8684123354677499 valid acc 16/16
Epoch 19 loss 1.1447229263710297 valid acc 15/16
Epoch 19 loss 0.574499600373649 valid acc 16/16
Epoch 19 loss 0.5364240146059036 valid acc 16/16
Epoch 19 loss 1.1800164554668249 valid acc 16/16
Epoch 20 loss 0.056600439526606316 valid acc 16/16
Epoch 20 loss 1.1498464450823567 valid acc 16/16
Epoch 20 loss 1.0247908363084046 valid acc 16/16
Epoch 20 loss 0.7445652151361297 valid acc 16/16
Epoch 20 loss 1.1733762574889082 valid acc 15/16
Epoch 20 loss 0.8073009808035319 valid acc 16/16
Epoch 20 loss 1.2976575484418567 valid acc 16/16
Epoch 20 loss 2.052354941739618 valid acc 16/16
Epoch 20 loss 1.8657871453814863 valid acc 16/16
Epoch 20 loss 0.704282092318621 valid acc 14/16
Epoch 20 loss 0.6424290276986253 valid acc 15/16
Epoch 20 loss 1.0024759842619277 valid acc 16/16
Epoch 20 loss 1.1920546133492294 valid acc 16/16
Epoch 20 loss 1.3409451018338685 valid acc 16/16
Epoch 20 loss 1.0204492108657095 valid acc 15/16
Epoch 20 loss 0.7750911221819361 valid acc 15/16
Epoch 20 loss 1.661338445597685 valid acc 16/16
Epoch 20 loss 1.764711761187171 valid acc 16/16
Epoch 20 loss 1.1402039085327593 valid acc 15/16
Epoch 20 loss 0.8938920467221346 valid acc 16/16
Epoch 20 loss 0.9323247690044607 valid acc 16/16
Epoch 20 loss 0.26365320253383295 valid acc 16/16
Epoch 20 loss 0.09145528775745682 valid acc 16/16
Epoch 20 loss 0.5879769671605493 valid acc 16/16
Epoch 20 loss 0.57939548141141 valid acc 16/16
Epoch 20 loss 1.019005625607614 valid acc 16/16
Epoch 20 loss 0.6721600589667407 valid acc 16/16
Epoch 20 loss 0.2976055368377301 valid acc 15/16
Epoch 20 loss 0.6848398807013438 valid acc 14/16
Epoch 20 loss 0.391746449933109 valid acc 16/16
Epoch 20 loss 0.7234362663069047 valid acc 16/16
Epoch 20 loss 0.8707318444472003 valid acc 15/16
Epoch 20 loss 0.9451850866388848 valid acc 16/16
Epoch 20 loss 0.9203263464441624 valid acc 16/16
Epoch 20 loss 0.9153241087563536 valid acc 16/16
Epoch 20 loss 0.7507643555658006 valid acc 16/16
Epoch 20 loss 0.4643062490851463 valid acc 16/16
Epoch 20 loss 0.3719880098240555 valid acc 16/16
Epoch 20 loss 0.9630665648080338 valid acc 16/16
Epoch 20 loss 0.661810289934505 valid acc 16/16
Epoch 20 loss 0.3951864381619426 valid acc 16/16
Epoch 20 loss 0.8155084131570127 valid acc 16/16
Epoch 20 loss 0.4076399347545174 valid acc 16/16
Epoch 20 loss 0.5268108468236331 valid acc 14/16
Epoch 20 loss 1.1140475443749756 valid acc 15/16
Epoch 20 loss 0.3655538092697451 valid acc 15/16
Epoch 20 loss 1.1122485860024367 valid acc 16/16
Epoch 20 loss 0.7615853930114322 valid acc 15/16
Epoch 20 loss 0.7590086742909956 valid acc 15/16
Epoch 20 loss 0.7138233176199992 valid acc 16/16
Epoch 20 loss 0.6207679683923071 valid acc 16/16
Epoch 20 loss 0.8817105795904202 valid acc 16/16
Epoch 20 loss 0.35946558624370906 valid acc 16/16
Epoch 20 loss 0.34637111030183426 valid acc 15/16
Epoch 20 loss 0.9437113645661688 valid acc 16/16
Epoch 20 loss 0.7776271587514652 valid acc 16/16
Epoch 20 loss 1.0107728444519923 valid acc 16/16
Epoch 20 loss 0.8464812736840384 valid acc 16/16
Epoch 20 loss 0.5050538298553496 valid acc 16/16
Epoch 20 loss 1.0890029724548427 valid acc 15/16
Epoch 20 loss 0.5738354192802452 valid acc 16/16
Epoch 20 loss 0.354771028739331 valid acc 16/16
Epoch 20 loss 0.9862169112919001 valid acc 16/16