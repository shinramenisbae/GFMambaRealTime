try:
    import mamba_ssm
    print('OK')
except Exception as e:
    print('ERR', repr(e))
