"""

- commit conditions 
1. updated source (.go, makefile)

> commit list
for commit in list:
    run test
    parse test output
    VVVVV
    output -> commit1.json

"""

[
    "f97a5920a89ab6060b49c823007c47d811f4d050": {
        'TestJsonNextValueBytes': {
            'result': False,
            'stdout': "codec_test.go:444: recovered error: runtime error: invalid memory address or nil pointer dereference", 
            'stacktrace': "...",
            'test_code': """
func TestJsonNextValueBytes(t *testing.T) {
	doTestNextValueBytes(t, testJsonH)
}

func doTestNextValueBytes(t *testing.T, h Handle) {
	defer testSetup(t)()

	bh := testBasicHandle(h)

	// - encode uint, int, float, bool, struct, map, slice, string - all separated by nil
	// - use nextvaluebytes to grab he's got each one, and decode it, and compare
	var inputs = []interface{}{
		uint64(7777),
		int64(9999),
		float64(12.25),
		true,
		false,
		map[string]uint64{"1": 1, "22": 22, "333": 333, "4444": 4444},
		[]string{"1", "22", "333", "4444"},
		// use *TestStruc, not *TestStrucFlex, as *TestStrucFlex is harder to compare with deep equal
		// Remember: *TestStruc was separated for this reason, affording comparing against other libraries
		newTestStruc(testDepth, testNumRepeatString, false, false, true),
		"1223334444",
	}
	var out []byte

	for i, v := range inputs {
		_ = i
		bs := testMarshalErr(v, h, t, "nextvaluebytes")
		out = append(out, bs...)
		bs2 := testMarshalErr(nil, h, t, "nextvaluebytes")
		out = append(out, bs2...)
		testReleaseBytes(bs)
		testReleaseBytes(bs2)
	}
	// out = append(out, []byte("----")...)

	var valueBytes = make([][]byte, len(inputs)*2)

	d, oldReadBufferSize := testSharedCodecDecoder(out, h, testBasicHandle(h))
	for i := 0; i < len(inputs)*2; i++ {
		valueBytes[i] = d.d.nextValueBytes([]byte{})
		// bs := d.d.nextValueBytes([]byte{})
		// valueBytes[i] = make([]byte, len(bs))
		// copy(valueBytes[i], bs)
	}
	if testUseIoEncDec >= 0 {
		bh.ReaderBufferSize = oldReadBufferSize
	}

	defer func(b bool) { bh.InterfaceReset = b }(bh.InterfaceReset)
	bh.InterfaceReset = false

	var result interface{}
	for i := 0; i < len(inputs); i++ {
		// result = reflect.New(reflect.TypeOf(inputs[i])).Elem().Interface()
		result = reflect.Zero(reflect.TypeOf(inputs[i])).Interface()
		testUnmarshalErr(&result, valueBytes[i*2], h, t, "nextvaluebytes")
		testDeepEqualErr(inputs[i], result, t, "nextvaluebytes-1")
		result = nil
		testUnmarshalErr(&result, valueBytes[(i*2)+1], h, t, "nextvaluebytes")
		testDeepEqualErr(nil, result, t, "nextvaluebytes-2")
	}
}
            """
        },
        ...
    }
]