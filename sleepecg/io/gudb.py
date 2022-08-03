# © SleepECG developers
#
# License: BSD (3-clause)

"""Utilities for downloading GUDB data."""

from pathlib import Path
from typing import Optional, Union

from ..config import get_config
from .utils import _calculate_checksum

GUDB_MD5 = {
    "subject_00/sitting/ECG.tsv": "06cdda76a17ff6efe4d4b18ccbc66e0f",
    "subject_00/sitting/annotation_cs.tsv": "fd0e6f5796a103637e29c4f9c14d155c",
    "subject_00/sitting/annotation_cables.tsv": "f5dfb5b4073974cc6aaf8ca3cc262002",
    "subject_00/maths/ECG.tsv": "c45d6702d76cb4083592b2c5ff2a298f",
    "subject_00/maths/annotation_cs.tsv": "839a5708475774a3a8ab6d7f31ac1646",
    "subject_00/maths/annotation_cables.tsv": "c61bf3a8b4174398ddaa04051889c62f",
    "subject_00/walking/ECG.tsv": "f690901d1868969a4dfb7a15c3dfa1cb",
    "subject_00/walking/annotation_cs.tsv": "37f4a8d5ba5722c1ce6cd2fc928a7740",
    "subject_00/walking/annotation_cables.tsv": "b88acd4f38d860dadebb522189114ce1",
    "subject_00/hand_bike/ECG.tsv": "9aa72a016ef41d8f4f39ad3689b9871a",
    "subject_00/hand_bike/annotation_cs.tsv": "881be7616a7c46b6b143739049db6c72",
    "subject_00/hand_bike/annotation_cables.tsv": "7db81d70dcbfc7cced14aa158e96470c",
    "subject_00/jogging/ECG.tsv": "11ed52586cb91ef185c7160905017ec2",
    "subject_00/jogging/annotation_cs.tsv": "762eb146f31a7ef487387cf9fe169bab",
    "subject_00/jogging/annotation_cables.tsv": "70f881a46e7818a7162a6effe6604207",
    "subject_01/sitting/ECG.tsv": "60d3a38e8a6b48c4a32d0b9a39277ef8",
    "subject_01/sitting/annotation_cs.tsv": "a4b432880debfb90958a48017d8f73f9",
    "subject_01/sitting/annotation_cables.tsv": "52a353522ad7c7a827c2daa5e9f7de89",
    "subject_01/maths/ECG.tsv": "abd07840c9f2579a8e4514d23c1373ec",
    "subject_01/maths/annotation_cs.tsv": "f93032d8893fedb88cf39906f49f0607",
    "subject_01/maths/annotation_cables.tsv": "ea9a1ade57a77c1d390e799e2dcc4445",
    "subject_01/walking/ECG.tsv": "5561332094a52070a8514d5fd81797c4",
    "subject_01/walking/annotation_cs.tsv": "2b3825df549347ba2c31ea142a941bb4",
    "subject_01/walking/annotation_cables.tsv": "98d554347efa2ec1927e1afe40443fd5",
    "subject_01/hand_bike/ECG.tsv": "d8f02a9e78e6e9bc21b54b6cf0597c33",
    "subject_01/hand_bike/annotation_cs.tsv": "6e0795c51df894e9b15a21b59ff2a3f4",
    "subject_01/hand_bike/annotation_cables.tsv": "f2f9261999fd8641eb04cbcc41123cd2",
    "subject_01/jogging/ECG.tsv": "e57e7e16162a09bf9a6efb1793bf5c7f",
    "subject_01/jogging/annotation_cs.tsv": "1fd3a1da946a321ac38b9cbd07efe329",
    "subject_02/sitting/ECG.tsv": "3d146c1d17276e69be76246a8da05c7b",
    "subject_02/sitting/annotation_cs.tsv": "5041b24aafdc6943ceeac17d7220f5e0",
    "subject_02/sitting/annotation_cables.tsv": "ac489881c4c675f453a90516356c6ddc",
    "subject_02/maths/ECG.tsv": "6f8c58cef3bc4a95fe2685010fb41f5a",
    "subject_02/maths/annotation_cs.tsv": "5d4b9d38d30d5b5b109b111b19f20cbf",
    "subject_02/maths/annotation_cables.tsv": "7c245283a2328e31f26cc6f408c10cf3",
    "subject_02/walking/ECG.tsv": "437999f0829a518a646f931c2f1bd7bd",
    "subject_02/walking/annotation_cs.tsv": "13255fe3c52683d351f83722bd344073",
    "subject_02/hand_bike/ECG.tsv": "e9619e9b8b4f7f27e1661688066d55ff",
    "subject_02/hand_bike/annotation_cables.tsv": "75a911a319c716191fcd19fb3c2f9f7e",
    "subject_02/jogging/ECG.tsv": "84328c7b013a69a30861eeffc7d7d111",
    "subject_02/jogging/annotation_cs.tsv": "7f4519ff3506fb8a13dbb7c2ad7b95f6",
    "subject_03/sitting/ECG.tsv": "0ce0b22352f55f5bef52278c8e4335fb",
    "subject_03/sitting/annotation_cs.tsv": "210a90fa594a5d2eb159fa781bad0869",
    "subject_03/sitting/annotation_cables.tsv": "9de91a8972cb065211a939c529abb04c",
    "subject_03/maths/ECG.tsv": "5ffecb8356436bd8a78f524618b5ca75",
    "subject_03/maths/annotation_cs.tsv": "dcca8a59145a65c2a00a59c78e384033",
    "subject_03/maths/annotation_cables.tsv": "f26f651f7ee086d724aabbf0940cebd1",
    "subject_03/walking/ECG.tsv": "fceb4b443b313366d51eccccdb60e056",
    "subject_03/walking/annotation_cs.tsv": "504801c9b7ff119403e4988b368264f8",
    "subject_03/walking/annotation_cables.tsv": "68b37b7de6b188fb03503b10b3f1e15c",
    "subject_03/hand_bike/ECG.tsv": "6242fd94978a0f33b19b1d0efd6bd22c",
    "subject_03/hand_bike/annotation_cs.tsv": "7be093d8f5508f4ea6f7875376433364",
    "subject_03/hand_bike/annotation_cables.tsv": "c633d0111a3f21d9e2c082172fbab846",
    "subject_03/jogging/ECG.tsv": "adf088c525a87e3b6f8de37846b1b343",
    "subject_03/jogging/annotation_cs.tsv": "c872437eae0ebd6019190faa7003f544",
    "subject_04/sitting/ECG.tsv": "3ee5cc35f6255b3e947e5b9972e7dd4c",
    "subject_04/sitting/annotation_cs.tsv": "fb6d3c272a3fa7ef4ca98351187eeebe",
    "subject_04/sitting/annotation_cables.tsv": "adf2b52bfbaa8b568f59259ea76a4764",
    "subject_04/maths/ECG.tsv": "3e7896ab35e71fb25a9620413e4bca02",
    "subject_04/maths/annotation_cs.tsv": "8003c1c616724a225269701937c37720",
    "subject_04/walking/ECG.tsv": "4defbae396be495c3a96ea9c59cddb25",
    "subject_04/walking/annotation_cs.tsv": "57dce829e0d490ff7e5ab2ddf4ac2461",
    "subject_04/walking/annotation_cables.tsv": "8090a8e51b002d1c8151ec15ca1c61ba",
    "subject_04/hand_bike/ECG.tsv": "96d130f8349bfc3258eee5a066c46ce9",
    "subject_04/hand_bike/annotation_cs.tsv": "e12e2cb46dc84cb4a75ad52e7b64e91a",
    "subject_04/jogging/ECG.tsv": "6e7737946cdff13020538ede6814ff78",
    "subject_04/jogging/annotation_cs.tsv": "bc807c4e439d0e435e06e413077af5ff",
    "subject_05/sitting/ECG.tsv": "0404357fc96a34b3df60984a61529d49",
    "subject_05/sitting/annotation_cs.tsv": "e026c6667435c12c660e9dac7e5e775d",
    "subject_05/sitting/annotation_cables.tsv": "21ae7fb00b6c20ee75ef7d5d0628b563",
    "subject_05/maths/ECG.tsv": "9f9f05f0d3635277de041d1bbf1e70cb",
    "subject_05/maths/annotation_cs.tsv": "3105881f3f49d10ac615da0eed15e8ce",
    "subject_05/maths/annotation_cables.tsv": "fabceb0904d94a4547fd53f9af381526",
    "subject_05/walking/ECG.tsv": "cfb25c7760bff6c25a9a6fb883939170",
    "subject_05/walking/annotation_cs.tsv": "e030781bcbd24cca222aa87146650b51",
    "subject_05/walking/annotation_cables.tsv": "c19a96336c2a19a6237099aea59ac366",
    "subject_05/hand_bike/ECG.tsv": "3f435bdbe8a434156c667da5c9fff564",
    "subject_05/hand_bike/annotation_cs.tsv": "da34e6a473dfa3cd2b5c1d4cd4a3fc9f",
    "subject_05/hand_bike/annotation_cables.tsv": "19274eddb7bacdf23805c982991bc53e",
    "subject_05/jogging/ECG.tsv": "eda97396628d68069d6bdc0cf585d00f",
    "subject_05/jogging/annotation_cs.tsv": "b0141b34977164fd4cea1f1729750f51",
    "subject_06/sitting/ECG.tsv": "68c7af8df45fc9fdd0e98402208ca6d3",
    "subject_06/sitting/annotation_cs.tsv": "a2dc6189c3df152e188ef1b54319504d",
    "subject_06/sitting/annotation_cables.tsv": "e1ff9a47d9e41e65eb3bcc743ea3cbbb",
    "subject_06/maths/ECG.tsv": "b149f71798a5fb074da17c91de732b03",
    "subject_06/maths/annotation_cs.tsv": "fa5e392e4d2524a2c384b0c57099a65a",
    "subject_06/maths/annotation_cables.tsv": "2599448b0cb7b9455dc1a464bff2a2a7",
    "subject_06/walking/ECG.tsv": "00e75ef5829bbdd113f110a340ebffac",
    "subject_06/walking/annotation_cs.tsv": "8b1b3316401de021845ee2ce37d1d656",
    "subject_06/walking/annotation_cables.tsv": "0b19d0c66b8c5eb5c8eda5483006fb5c",
    "subject_06/hand_bike/ECG.tsv": "eff3769417ebc8550b2ea5f9617ec158",
    "subject_06/hand_bike/annotation_cs.tsv": "b09aa1f9c82b11223ea380b83861b943",
    "subject_06/hand_bike/annotation_cables.tsv": "c2a0d9bd07ffe7a7bb9d7893a0c499ff",
    "subject_06/jogging/ECG.tsv": "a1995ab4a6cefdf530e5eab89cace474",
    "subject_06/jogging/annotation_cs.tsv": "0d1d60adb1c5bf67c2165ec6470e229f",
    "subject_06/jogging/annotation_cables.tsv": "f08d596e1edb02d0db7965965a054400",
    "subject_07/sitting/ECG.tsv": "342e6889e0603edf0956ba2067742a77",
    "subject_07/sitting/annotation_cs.tsv": "f4659f452164b1c34d49bd0b9ff49ea2",
    "subject_07/sitting/annotation_cables.tsv": "42e865caa040999054c7910cb58b73e0",
    "subject_07/maths/ECG.tsv": "e2952fd5478aa6987868a537e1717298",
    "subject_07/maths/annotation_cs.tsv": "4797a326c46d48e2847f1216c796d3e7",
    "subject_07/maths/annotation_cables.tsv": "4e2dda2006496f58550a45fcc09e664f",
    "subject_07/walking/ECG.tsv": "5514687e33e0dd67bcf0d8c11a1a72f7",
    "subject_07/walking/annotation_cs.tsv": "22e561270358bbef887637e0af334225",
    "subject_07/walking/annotation_cables.tsv": "319b13c3822c21b42b65ffdf86f32600",
    "subject_07/hand_bike/ECG.tsv": "86980328f6c96a6381371c1a4fc951c8",
    "subject_07/hand_bike/annotation_cs.tsv": "e52a73e3e8919607bed9779d649bc056",
    "subject_07/hand_bike/annotation_cables.tsv": "2f8d7de022798473b18137c1fe3e0072",
    "subject_07/jogging/ECG.tsv": "e9a064b387e7b67dd59177d0f7e292e7",
    "subject_07/jogging/annotation_cs.tsv": "a88ef36a1d60fce4802341e35dc788fe",
    "subject_07/jogging/annotation_cables.tsv": "5ed96ba00aeb3aff4787e88e379e554a",
    "subject_08/sitting/ECG.tsv": "ebe626cae85178fd0b52869bdcf7eb99",
    "subject_08/sitting/annotation_cs.tsv": "ed5bb1f08eb18058becda4ddff267fd8",
    "subject_08/sitting/annotation_cables.tsv": "f01167a2f603964be6bf005235bb86a7",
    "subject_08/maths/ECG.tsv": "1ad476ef5ef1058e766441ee85d8ab0e",
    "subject_08/maths/annotation_cs.tsv": "55f3dd297d818bcd3d4d07b2dd4dd1db",
    "subject_08/maths/annotation_cables.tsv": "e352b8c3973cf9dbec3453fc2df5c0da",
    "subject_08/walking/ECG.tsv": "a47892bc5f9895ec61530f90993715be",
    "subject_08/walking/annotation_cs.tsv": "bd0ae6b87f46dac06daca10241edc24c",
    "subject_08/walking/annotation_cables.tsv": "1b316f0278ce639cd54ea3fbe7459604",
    "subject_08/hand_bike/ECG.tsv": "4c904204b830e65baf9878b5d4d8d0cd",
    "subject_08/hand_bike/annotation_cs.tsv": "0d01dcc8d1a2b69e8f58c45d0b33646f",
    "subject_08/hand_bike/annotation_cables.tsv": "4c3ce44963dc9a909285e6c6f54f1557",
    "subject_08/jogging/ECG.tsv": "62ca57c851a269e8068bde0d4b0fd3e9",
    "subject_08/jogging/annotation_cs.tsv": "a471bd10cd03189e222743343a053db7",
    "subject_08/jogging/annotation_cables.tsv": "5ba5293989be79509212b216b9548c79",
    "subject_09/sitting/ECG.tsv": "d7191fdc7bbfc8d86e78fd2029f1e556",
    "subject_09/sitting/annotation_cs.tsv": "6ddcb613e1950b9d6f424f2d275e8309",
    "subject_09/sitting/annotation_cables.tsv": "94e0394b1d539c1c6f88597fcaa66bba",
    "subject_09/maths/ECG.tsv": "4ff7380d04e94d84a15eac7431c6e06a",
    "subject_09/maths/annotation_cs.tsv": "94a7e2617be9bf0231f9663898ef3774",
    "subject_09/maths/annotation_cables.tsv": "c09ce801d1c1e15504385e541a0b431b",
    "subject_09/walking/ECG.tsv": "2e027f36931458a39af1be831d1119d2",
    "subject_09/walking/annotation_cs.tsv": "fa358656c678376a4104a1ff1f0e78f1",
    "subject_09/walking/annotation_cables.tsv": "d19311e6e929b56ad1718a31d0f4abfc",
    "subject_09/hand_bike/ECG.tsv": "bc66594febe813dbdea1cac6ed84274c",
    "subject_09/hand_bike/annotation_cs.tsv": "f3965cf50bb812000537638062163ffd",
    "subject_09/hand_bike/annotation_cables.tsv": "23970f6baaeaba487a177463a599380b",
    "subject_09/jogging/ECG.tsv": "1ac4a47cf58644f9fc974ff8073b5eb8",
    "subject_09/jogging/annotation_cs.tsv": "3f480d90be9e2eaa734f8c1bc20c6612",
    "subject_10/sitting/ECG.tsv": "451fba00370d0f55798b15924d08db2b",
    "subject_10/sitting/annotation_cs.tsv": "c615a79a5287f6e10bf02745709ca8d0",
    "subject_10/sitting/annotation_cables.tsv": "448e48ed2d8b1be35738ed2a141732bc",
    "subject_10/maths/ECG.tsv": "7a698d640a3d248794a6b49f3eee0a7d",
    "subject_10/maths/annotation_cs.tsv": "97fa8866c103b7fed0ccfe86f3fd9cec",
    "subject_10/maths/annotation_cables.tsv": "e4b45f40698c5685c7099b74d1953a4a",
    "subject_10/walking/ECG.tsv": "65dd73737d08a7bf53290c6b4e81c9a8",
    "subject_10/walking/annotation_cs.tsv": "fa575510c1b407df09db89ddd6b38663",
    "subject_10/walking/annotation_cables.tsv": "ad56de0cfe9f30f7f3aa814dcb34f4bd",
    "subject_10/hand_bike/ECG.tsv": "c229c9d5899ba4c5a563e94b838d6256",
    "subject_10/hand_bike/annotation_cs.tsv": "443445d27c64c1f7515429e0c944debb",
    "subject_10/hand_bike/annotation_cables.tsv": "4dbabe9f8e1b8d10ed47c67da6222dba",
    "subject_10/jogging/ECG.tsv": "529ed6542ea5d713688530c813ee141e",
    "subject_10/jogging/annotation_cs.tsv": "88a87727f517cb674ed8971b76107589",
    "subject_11/sitting/ECG.tsv": "5576b38f597db07a4507c1191274c780",
    "subject_11/sitting/annotation_cs.tsv": "fd5e0fb304482430c6d61db460fbbeb6",
    "subject_11/sitting/annotation_cables.tsv": "166a5358834a285d6ad176598dad44af",
    "subject_11/maths/ECG.tsv": "1fa97be10d4359842f299ee75de0de63",
    "subject_11/maths/annotation_cs.tsv": "089191183e5449c21da485b15d2d8e14",
    "subject_11/maths/annotation_cables.tsv": "9f322f27b2ae4d04746c0f5dad3b2d50",
    "subject_11/walking/ECG.tsv": "1483ee2c3052e3e64fa4754c3b0e6701",
    "subject_11/walking/annotation_cs.tsv": "8a7004a5b11be8e31429c2e107694159",
    "subject_11/walking/annotation_cables.tsv": "87ae6985edcdc3d3895439aa39f29c67",
    "subject_11/hand_bike/ECG.tsv": "df44a2bc9bfdeb336531ecb1239e0bbc",
    "subject_11/hand_bike/annotation_cs.tsv": "0c22d78d70829ab0a4516b250464d31c",
    "subject_11/hand_bike/annotation_cables.tsv": "05a909caa55815539e2663cd6f4e15f8",
    "subject_11/jogging/ECG.tsv": "f718fee2650ee52109cd8b58f03d2b69",
    "subject_11/jogging/annotation_cs.tsv": "296c20c583fbc88b8de07362cf504583",
    "subject_12/sitting/ECG.tsv": "8fbe2a782559dd610e0abc5264c50fd8",
    "subject_12/sitting/annotation_cs.tsv": "39c70407af75b8faf75a567cc6a46348",
    "subject_12/sitting/annotation_cables.tsv": "466ba8c02493c74fde1f326c5b0e0bf2",
    "subject_12/maths/ECG.tsv": "a7111e8cfd8371776ebd071a875f9daa",
    "subject_12/maths/annotation_cs.tsv": "a49d8f7e71e691be4ab543760bf74749",
    "subject_12/maths/annotation_cables.tsv": "9260e4f007dccfa97e30d0918accf3c4",
    "subject_12/walking/ECG.tsv": "616c5d2b992373c9c73bf7560bae4a9f",
    "subject_12/walking/annotation_cs.tsv": "c390d23e94e1090801d00f425250cb6d",
    "subject_12/walking/annotation_cables.tsv": "275bda8be776576c89ea178dcfd762c5",
    "subject_12/hand_bike/ECG.tsv": "01857ddeac3b1a5c50475aea13a068cb",
    "subject_12/hand_bike/annotation_cs.tsv": "0a2bdc85664c036e537b64f0d65d5869",
    "subject_12/hand_bike/annotation_cables.tsv": "9017c806f9e198081c50024de70eb8c5",
    "subject_12/jogging/ECG.tsv": "17645a0761ff19abd1a6cbf7e4f019bc",
    "subject_12/jogging/annotation_cs.tsv": "b32954d2ae722fcba3eb4870e4600900",
    "subject_12/jogging/annotation_cables.tsv": "519c7b20aca9ab766a6af6e53e61f8f2",
    "subject_13/sitting/ECG.tsv": "2b8f8bb2f9a6ea10c759f2cf7dd5f52c",
    "subject_13/sitting/annotation_cs.tsv": "8ced97828c40d3210e3912051bc1e9fd",
    "subject_13/sitting/annotation_cables.tsv": "81940a457136793394c3dce09aa15770",
    "subject_13/maths/ECG.tsv": "684e0c4da0668643ba3e2601acd89a26",
    "subject_13/maths/annotation_cs.tsv": "62320f5eb5f4fe7631707d019b184122",
    "subject_13/maths/annotation_cables.tsv": "c5375d6142a08137ee26e828e4caaefd",
    "subject_13/walking/ECG.tsv": "562a736aaabdc2e2f0e3023127d078e7",
    "subject_13/walking/annotation_cs.tsv": "62c000d40edc64a47208776795500c30",
    "subject_13/walking/annotation_cables.tsv": "0641c7bb3640216e7db64d95eb445447",
    "subject_13/hand_bike/ECG.tsv": "c5026cd15178e02adcbb720933a42bc8",
    "subject_13/hand_bike/annotation_cs.tsv": "5a38e982f54bb3d7657d0e361004853f",
    "subject_13/hand_bike/annotation_cables.tsv": "c3828b757ecf31ec7dd32a575b1633ce",
    "subject_13/jogging/ECG.tsv": "2ab3b4981e230abf1fc1afe71d41685a",
    "subject_13/jogging/annotation_cs.tsv": "8c25781ceba1de40c4bad5d4490061e0",
    "subject_13/jogging/annotation_cables.tsv": "7a510ec1fb4efc3e3516d393625dc6f9",
    "subject_14/sitting/ECG.tsv": "d4ddc2d8c6798ea9801a056ee7049836",
    "subject_14/sitting/annotation_cs.tsv": "46efc93ae3292620a31ebbb36938d704",
    "subject_14/sitting/annotation_cables.tsv": "a5f73554ce1b49991da4267d9368192e",
    "subject_14/maths/ECG.tsv": "e7b56f7af8be15986ac09bca9bce1e3b",
    "subject_14/maths/annotation_cs.tsv": "676e68e0b1990b1b70bcc7e83ed72261",
    "subject_14/maths/annotation_cables.tsv": "ba1b80a5f2725e41d35bb9780472eb22",
    "subject_14/walking/ECG.tsv": "1c7b2d14433aa3aabb502793dbfaad9a",
    "subject_14/walking/annotation_cs.tsv": "40b06dcd21c54cf67420ab70b26ed21f",
    "subject_14/walking/annotation_cables.tsv": "40aae4c0473d154c0ba20a4d57f43181",
    "subject_14/hand_bike/ECG.tsv": "df00a34c892bf65bc435a7a26b711b74",
    "subject_14/hand_bike/annotation_cs.tsv": "66664ec44564e3732a8617e46ecfba50",
    "subject_14/hand_bike/annotation_cables.tsv": "8a1def22be44606cf241e1657084c021",
    "subject_14/jogging/ECG.tsv": "488c40df09bf479d49a5080431523909",
    "subject_15/sitting/ECG.tsv": "ff58a363d78a32e027ee4cf82c21276d",
    "subject_15/sitting/annotation_cs.tsv": "ccd58e85e755fcdf34cf2aac6d8a2c3f",
    "subject_15/sitting/annotation_cables.tsv": "1e560b2f3005e715451f80ab4d00ab4b",
    "subject_15/maths/ECG.tsv": "a61b21865e895aac969405e4f9334067",
    "subject_15/maths/annotation_cs.tsv": "5a123fa6ed029a8b074096b488ca5539",
    "subject_15/maths/annotation_cables.tsv": "8fef825d8b5b3df847b117603144f914",
    "subject_15/walking/ECG.tsv": "d2d8c2a0df608b13122813a05a4f882d",
    "subject_15/walking/annotation_cs.tsv": "d802bc5615af97800e02963c1f7711bc",
    "subject_15/walking/annotation_cables.tsv": "13376400aca9fb3de99d0e8c7f42df9f",
    "subject_15/hand_bike/ECG.tsv": "b39a2b6622346d5784c8eb1ba5c033fb",
    "subject_15/hand_bike/annotation_cs.tsv": "92b803d82f57f56d4c50dc6e5e0793c7",
    "subject_15/hand_bike/annotation_cables.tsv": "75cde32ab90fac283bbc94deeeda1611",
    "subject_15/jogging/ECG.tsv": "d964156f041c1500bf4a8c6c1c67f642",
    "subject_15/jogging/annotation_cs.tsv": "63bd3eaeb4754e4f873242dc8bedb36f",
    "subject_16/sitting/ECG.tsv": "6f1db64ff99add2619d1177755436d62",
    "subject_16/sitting/annotation_cs.tsv": "93b75b892dfe9f3cc37c58a4ad48a304",
    "subject_16/sitting/annotation_cables.tsv": "f3238d2e71f2390875688a4cbf358c6e",
    "subject_16/maths/ECG.tsv": "656df44b9dbcd448fd8343b7c1e3e67e",
    "subject_16/maths/annotation_cs.tsv": "26e7cb2421e1b980cf5599369790119f",
    "subject_16/maths/annotation_cables.tsv": "d23d78b371e7e62ae07ab4471e7877d2",
    "subject_16/walking/ECG.tsv": "b24b3accaf21602917a9154686af7bc8",
    "subject_16/walking/annotation_cs.tsv": "0cb17c2a6f4ef39d74625dc4d4bfc07f",
    "subject_16/walking/annotation_cables.tsv": "98bab2156283ae72faa6c6a96be17c4e",
    "subject_16/hand_bike/ECG.tsv": "25e3d6dfccabdb80af973c52721d39ca",
    "subject_16/hand_bike/annotation_cs.tsv": "3eb7052c00ccaebc30a82f6af1d75f96",
    "subject_16/hand_bike/annotation_cables.tsv": "5ecf1a6a83522e7668c0288f59803267",
    "subject_16/jogging/ECG.tsv": "12270581c1712ddcc919299aeb5d3cbc",
    "subject_16/jogging/annotation_cs.tsv": "9433b94232a25c1361d448ba8e821c9f",
    "subject_17/sitting/ECG.tsv": "d948e64174f604697ce535ff9945359b",
    "subject_17/sitting/annotation_cs.tsv": "77acf669444fbb6385bbe52a26f3b5e1",
    "subject_17/sitting/annotation_cables.tsv": "90de09c439aa46bab587c6ab0b7b9acd",
    "subject_17/maths/ECG.tsv": "d2c4e0f8ac8db5c008ef8fb52caa4101",
    "subject_17/maths/annotation_cs.tsv": "12c24101f99eb6ad98a7a34036e10bde",
    "subject_17/maths/annotation_cables.tsv": "ed92d89a03c5cb2339d972d8cfd7393d",
    "subject_17/walking/ECG.tsv": "dcdec4fab73e935e5628493e8065772b",
    "subject_17/walking/annotation_cs.tsv": "9add42765d98727c063c9e46e191b1f8",
    "subject_17/walking/annotation_cables.tsv": "03b3308c045c552088ef433f6cd98b64",
    "subject_17/hand_bike/ECG.tsv": "716eddb07963411051f682afb956f948",
    "subject_17/hand_bike/annotation_cs.tsv": "7836ed22698da8e06e16d378d16c5dfe",
    "subject_17/hand_bike/annotation_cables.tsv": "f2d9e1e7e8598a8519a0052c0b9b95a4",
    "subject_17/jogging/ECG.tsv": "32445d93c052f8eea18fd2f5dbe18002",
    "subject_17/jogging/annotation_cs.tsv": "5326675c5186c92bf0c96af7465f1580",
    "subject_17/jogging/annotation_cables.tsv": "c1a82769e3c22965d4ebcb62ddb1c343",
    "subject_18/sitting/ECG.tsv": "be5d9766d5c62a447160120ea814be0e",
    "subject_18/sitting/annotation_cs.tsv": "bb1d94e053533a0372be8ad13e96bec2",
    "subject_18/sitting/annotation_cables.tsv": "e15e6efa08fc3d59a88790904872d56e",
    "subject_18/maths/ECG.tsv": "2a11444bc2a6c710ae2717bb5490941e",
    "subject_18/maths/annotation_cs.tsv": "1fe8d3e2b6c9bf6ebb2365d88c5c1a33",
    "subject_18/maths/annotation_cables.tsv": "93dc8aa4b5d9c72923f09207f9ba0276",
    "subject_18/walking/ECG.tsv": "69e3d082e9ff558910b3c3a69ca7db2d",
    "subject_18/walking/annotation_cs.tsv": "80a85c1f670788e63001746cad1f80e7",
    "subject_18/walking/annotation_cables.tsv": "259379dd5bab43699cf1b123138ac6b7",
    "subject_18/hand_bike/ECG.tsv": "bce3f9e3d0085926a5d0415343cd97e0",
    "subject_18/hand_bike/annotation_cs.tsv": "b4d2413bc7bc4054a9006ca4f8bd1524",
    "subject_18/hand_bike/annotation_cables.tsv": "234577f3c15488eb60dbee4f01dedf14",
    "subject_18/jogging/ECG.tsv": "b356d1907fd808bb70ec9542c0928965",
    "subject_18/jogging/annotation_cs.tsv": "55de2d7b1272b790bfcdf571029da561",
    "subject_18/jogging/annotation_cables.tsv": "95e9f78308c7e1f99a50fb626f87714e",
    "subject_19/sitting/ECG.tsv": "96d436a264d4b65e3e13b2fa02d3bde4",
    "subject_19/sitting/annotation_cs.tsv": "e6b1518cbce2e4b48895edc304b91de3",
    "subject_19/sitting/annotation_cables.tsv": "e052065d0d9f36fd8a119ae8b995c950",
    "subject_19/maths/ECG.tsv": "a48606b8b66c9562f358c418a58b8765",
    "subject_19/maths/annotation_cs.tsv": "6559974b468495c6f30b6968aa9571c1",
    "subject_19/maths/annotation_cables.tsv": "0b50ef8f179535a27e1286e0f94b5cc3",
    "subject_19/walking/ECG.tsv": "5912959fd37e3c95c08509bab829e587",
    "subject_19/walking/annotation_cs.tsv": "736109d3c6e7066cbe1025e106b8d922",
    "subject_19/walking/annotation_cables.tsv": "8a2ffae2b8812d1a561a07171110e001",
    "subject_19/hand_bike/ECG.tsv": "dd36b0ae2b1d38882902733e19c1380f",
    "subject_19/hand_bike/annotation_cs.tsv": "03fd21a0041cf715a00de64357f762f0",
    "subject_19/hand_bike/annotation_cables.tsv": "4c13ce22d25a2dd61b184b52a893636d",
    "subject_19/jogging/ECG.tsv": "f4911206a46c7d006e4c8c27a677b143",
    "subject_19/jogging/annotation_cs.tsv": "c70ab97c9dde5a7e652f78eb2c16fcb1",
    "subject_20/sitting/ECG.tsv": "2f03c0ee5c40309366b9c2a78b735c2c",
    "subject_20/sitting/annotation_cs.tsv": "6c3ad7f91649f7364fe3b7ad7417918a",
    "subject_20/sitting/annotation_cables.tsv": "64d87cd99f4b8e7f7d53fc9944cb2364",
    "subject_20/maths/ECG.tsv": "06aa5ef809efae9888e9e71e66dae2e5",
    "subject_20/maths/annotation_cs.tsv": "6e72e82e76d0c8aa83d072987f2334cb",
    "subject_20/maths/annotation_cables.tsv": "52a94ea3ab0b0474a2c45db960beb246",
    "subject_20/walking/ECG.tsv": "8bcbd6a0b8a971ed21ed2c80be1b28f1",
    "subject_20/walking/annotation_cs.tsv": "dc06b36539a63462bd9568364cfde73f",
    "subject_20/walking/annotation_cables.tsv": "c55da8c316bb234661d65416942d57c5",
    "subject_20/hand_bike/ECG.tsv": "42fe2f7fad7f42ffd9cc78caebdcbb85",
    "subject_20/hand_bike/annotation_cs.tsv": "1129fa10885ed6cb9ae85e5770c52eec",
    "subject_20/hand_bike/annotation_cables.tsv": "05c0a5da17f081fe14c5beabdae0949e",
    "subject_20/jogging/ECG.tsv": "8e30adc2cfd8dde00cb2c7b45b67941f",
    "subject_20/jogging/annotation_cs.tsv": "2a4b97beaa6dbe4b06008f4b3edbdd83",
    "subject_21/sitting/ECG.tsv": "ae868cd24ac72317f7a5a7375ca80fc5",
    "subject_21/sitting/annotation_cs.tsv": "589b2cfd3da0d15323fc9f2a8e187169",
    "subject_21/sitting/annotation_cables.tsv": "0d96438ea65bc10e9a3873cb1c76dd53",
    "subject_21/maths/ECG.tsv": "701595699c176456b2f13981ccd49152",
    "subject_21/maths/annotation_cs.tsv": "380fe4da2cbac72e2c8312a2baf07e7c",
    "subject_21/maths/annotation_cables.tsv": "8bfb5117771270842594be5ef65fd36b",
    "subject_21/walking/ECG.tsv": "97ef97809a1228d555f3787eb84d87fd",
    "subject_21/walking/annotation_cs.tsv": "e860f968a098b11231654d6349ca52ff",
    "subject_21/walking/annotation_cables.tsv": "52a6619e82d48847f06a48b8b8b9a212",
    "subject_21/hand_bike/ECG.tsv": "3e747930059c56229b8f74b2fc0a76b2",
    "subject_21/hand_bike/annotation_cs.tsv": "0a4e07ef1d7d4583e524de0f199504ba",
    "subject_21/hand_bike/annotation_cables.tsv": "0a1adaff3e2e1c2a01ad84e6078900ef",
    "subject_21/jogging/ECG.tsv": "b316b9772ef5436a1c5190f9c16653a4",
    "subject_21/jogging/annotation_cs.tsv": "3e438e80ea8c47ead7a4e42aed1e2712",
    "subject_22/sitting/ECG.tsv": "df9e3dc11b20795cdc7b58a158f28845",
    "subject_22/sitting/annotation_cs.tsv": "adea5c03551b49d39d3373a2ca295508",
    "subject_22/sitting/annotation_cables.tsv": "e306412536887cd430a96e1180f39312",
    "subject_22/maths/ECG.tsv": "2798360475652bd93edd6ac634965c36",
    "subject_22/maths/annotation_cs.tsv": "2d65fbfb2a619d21b274e599357e5911",
    "subject_22/maths/annotation_cables.tsv": "e7c48d0a854e8202b2c2944285405e77",
    "subject_22/walking/ECG.tsv": "4d170188553afdc22da9c45f2eb10ad3",
    "subject_22/walking/annotation_cs.tsv": "015156a76b1b95cd3a250987a0f285fb",
    "subject_22/walking/annotation_cables.tsv": "7864eb48a4b6ac15743b4cb07acb55d1",
    "subject_22/hand_bike/ECG.tsv": "04ae294453f02bc52ddbe2dd0d6eaa37",
    "subject_22/hand_bike/annotation_cs.tsv": "55a128bebb152f8033fa27dcd6f28e66",
    "subject_22/hand_bike/annotation_cables.tsv": "b4f1196fc93e4c386f4d621a21e804f0",
    "subject_22/jogging/ECG.tsv": "ce297aa927fc6da534d84990b852b57a",
    "subject_22/jogging/annotation_cs.tsv": "609d9ea5a2675acb85812c74d538802d",
    "subject_23/sitting/ECG.tsv": "7bf3b1869f2f902de06522ca9ff05e90",
    "subject_23/sitting/annotation_cs.tsv": "4852264b275c870adc258d05a1c2e757",
    "subject_23/sitting/annotation_cables.tsv": "7c27b9b2fbddc9bff9b15b07138f4376",
    "subject_23/maths/ECG.tsv": "46375f8d9272327c993e457274d99c25",
    "subject_23/maths/annotation_cs.tsv": "008b6bc4564476531adfedbf59706f5a",
    "subject_23/maths/annotation_cables.tsv": "2c9f6bcbcdafd316b6fbf855221454a8",
    "subject_23/walking/ECG.tsv": "052256151a597f216ae59a82fa6c2cbf",
    "subject_23/walking/annotation_cs.tsv": "8f675544e362eee34eb5384286315977",
    "subject_23/walking/annotation_cables.tsv": "52940e00688d3b4d7f5c0e5b75545915",
    "subject_23/hand_bike/ECG.tsv": "b36212f50e67a45d44947c8a8b506fdb",
    "subject_23/hand_bike/annotation_cs.tsv": "e971f8d469500adfc55327c571a1af59",
    "subject_23/hand_bike/annotation_cables.tsv": "e511e70439fbac0208991e0ea2862135",
    "subject_23/jogging/ECG.tsv": "88abdcfb612523500fd09ba99d03d031",
    "subject_23/jogging/annotation_cs.tsv": "aad1ac021992a1205e225ed353002419",
    "subject_24/sitting/ECG.tsv": "16395f11d73da0bd6ad625306696783a",
    "subject_24/sitting/annotation_cs.tsv": "3b2df2a840f4436ba5b4a8d6b63a6ff8",
    "subject_24/sitting/annotation_cables.tsv": "1f9341d70093daa1ea9c4f6b91ed1bdf",
    "subject_24/maths/ECG.tsv": "e30984dcd5a5cb3f0de3eb5edf0e1de9",
    "subject_24/maths/annotation_cs.tsv": "7ebee3f47e5e10eb6673d5ad10f5da22",
    "subject_24/maths/annotation_cables.tsv": "7fb335e07d450dc999f4a34f810c767a",
    "subject_24/walking/ECG.tsv": "ebc9f1e04caa0255b3ff4e0aec4af232",
    "subject_24/walking/annotation_cs.tsv": "3100a93998ed861a624f5fbc9ca0b745",
    "subject_24/walking/annotation_cables.tsv": "771e287dd47a742f478dc2e461a671a7",
    "subject_24/hand_bike/ECG.tsv": "d8398aa9e20f80353042494e634ecc88",
    "subject_24/hand_bike/annotation_cs.tsv": "6cf91e81f80c30c772d06bc43cbdd85b",
    "subject_24/hand_bike/annotation_cables.tsv": "0b138798b9dbe8661fa2727d5f2220e6",
    "subject_24/jogging/ECG.tsv": "6f91604784f258ac6ca8f7f9408a2e03",
    "subject_24/jogging/annotation_cs.tsv": "74226610ebd87d1a13ea015ad4b40d7f",
    "subject_24/jogging/annotation_cables.tsv": "05ce602e00716d40607f6bb5665fc0a4",
}


def _generate_gudb_md5(data_dir: Optional[Union[str, Path]] = None):
    """
    Compute checksums of files in GUDB.

    This function can be used to compute the checksums from scratch if the data is already
    available locally. The global `GUDB_MD5` dictionary should be equal to the return value
    of this function, so usually it is not necessary to run this function.

    Parameters
    ----------
    data_dir : str | pathlib.Path, optional
        Directory where all datasets are stored. If `None` (default), the value will be
        taken from the configuration.

    Returns
    -------
    dict
        Checksums for all files in GUDB.
    """
    EXPERIMENTS = ["sitting", "maths", "walking", "hand_bike", "jogging"]

    if data_dir is None:
        data_dir = get_config("data_dir")

    db_dir = Path(data_dir).expanduser() / "gudb"
    checksums = {}
    for subject_id in range(25):
        for experiment in EXPERIMENTS:
            experiment_subdir = f"subject_{subject_id:02}/{experiment}"
            for tsv_filename in ("ECG.tsv", "annotation_cs.tsv", "annotation_cables.tsv"):
                target_filepath = db_dir / experiment_subdir / tsv_filename
                try:
                    checksum = _calculate_checksum(target_filepath, "md5")
                except FileNotFoundError:
                    pass
                else:
                    checksums[str(Path(experiment_subdir) / tsv_filename)] = checksum
    return checksums
