""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

# iu_boundary应该在espeak-ng中被保留且不被发音，所以可以找与该语言无关的punctuation，¿使用在西语中，英文中不会出现
_letters_iu_boundary = '¿'   

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# 不加boudary symbol时
#symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")
