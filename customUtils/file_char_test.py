import os
import string

# 테스트 대상 문자 집합
classes = list(string.printable[:-6])
classes.append('EOS')
classes.append('PADDING')
classes.append('UNKNOWN')

# 테스트할 루트 경로 (임시 폴더 등으로 설정하세요)
base_path = "./test_safe_names"

os.makedirs(base_path, exist_ok=True)

for char in classes:
    try:
        # 문자 기반 폴더명 만들기 (문자가 공백이거나 특수 문자여도 시도)
        folder_name = f"char_{char}"
        test_path = os.path.join(base_path, folder_name)

        os.makedirs(test_path)
        print(f"✅ Success: '{char}' (ord: {ord(char) if len(char)==1 else 'N/A'})")
    except Exception as e:
        print(f"❌ Failed: '{char}' (ord: {ord(char) if len(char)==1 else 'N/A'}) - {e}")
