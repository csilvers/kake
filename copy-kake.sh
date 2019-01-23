#!/bin/sh

# Copy kake from the KA source tree to the opensource source tree.
# This does code-munging to deal with differences between the two repos.

cp "$HOME"/khan/webapp/build/kake/lib/*.py kake/
cp "$HOME"/khan/webapp/build/kake/make.py kake/
rm kake/compile_zip*
mv kake/*_test.py tests/
rm kake/testutil.py

perl -pli -e 's/from build.kake.lib import/from . import/' kake/*.py
perl -pli -e 's/import build\.kake\.lib\./from . import /' kake/*.py
perl -pli -e 's/build\.kake\.lib\.//g' kake/*.py

perl -pli -e 's/from shared import ka_root/from . import project_root/' kake/*.py
perl -pli -e 's/from shared import ka_root/from kake import project_root/' tests/*.py
perl -pli -e 's/ka_root/project_root/' kake/*.py tests/*.py

perl -nli -e 'print unless /shared\.hooks/' kake/*.py

perl -pli -e 's/^import cPickle/try:\n    import cPickle\nexcept ImportError:\n    import pickle as cPickle  # python3/' kake/*.py tests/*.py
perl -pli -e 's/^import mock/try:\n    from unittest import mock  # python 3\nexcept ImportError:\n    import mock/' tests/*.py

perl -pli -e 's/from build.kake.lib import/from kake import/' tests/*.py
perl -pli -e 's/import build\.kake\.lib\./from kake import /' tests/*.py
perl -pli -e 's/build\.kake\.lib\.//g' tests/*.py
perl -pli -e 's/from kake import testutil/import testutil/' tests/*.py

perl -pli -e 's/build\.kake\.make/kake.make/g' tests/*.py

perl -nli -e 'print unless /\btestsize\b/' tests/*.py
perl -nli -e 'print unless /\btestutil\.decorators\b/' tests/*.py

perl -pli -e 's/mock\.patch\((.)log\./mock.patch(\1kake.log./g' tests/*.py

# For python3 compat
perl -pli -e 's/\biteritems\b/items/' kake/*.py tests/*.py
perl -pli -e 's/\bitervalues\b/values/' kake/*.py tests/*.py
perl -pli -e 's/\bxrange\b/range/' kake/*.py tests/*.py
perl -pli -e 's/print >>([^,\s]*), ([^(].*)/\1.write(\2 + "\\n")/' kake/*.py tests/*.py
perl -pli -e 's/print >>([^,\s]*)([^,]*)$/\1.write("\\n")\2/' kake/*.py tests/*.py

for f in tests/*_test.py; do
    grep -q 'if __name__ == ' "$f" && continue
    echo >> "$f"
    echo >> "$f"
    echo "if __name__ == '__main__':" >> "$f"
    echo "    testutil.main()" >> "$f"
done

echo "--- Also see if you need to apply any diffs to tests/testutil.py"
