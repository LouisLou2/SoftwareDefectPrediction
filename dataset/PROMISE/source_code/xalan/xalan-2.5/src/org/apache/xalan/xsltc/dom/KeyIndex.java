/*
 * @(#)$Id: KeyIndex.java,v 1.11 2003/04/01 21:39:16 mkwan Exp $
 *
 * The Apache Software License, Version 1.1
 *
 *
 * Copyright (c) 2001-2003 The Apache Software Foundation.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution
 *
 * 3. The end-user documentation included with the redistribution,
 *    if any, must include the following acknowledgment:
 *       "This product includes software developed by the
 *        Apache Software Foundation (http://www.apache.org/)."
 *    Alternately, this acknowledgment may appear in the software itself,
 *    if and wherever such third-party acknowledgments normally appear.
 *
 * 4. The names "Xalan" and "Apache Software Foundation" must
 *    not be used to endorse or promote products derived from this
 *    software without prior written permission. For written
 *    permission, please contact apache@apache.org.
 *
 * 5. Products derived from this software may not be called "Apache",
 *    nor may "Apache" appear in their name, without prior written
 *    permission of the Apache Software Foundation.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation and was
 * originally based on software copyright (c) 2001, Sun
 * Microsystems., http://www.sun.com.  For more
 * information on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 *
 * @author Morten Jorgensen
 * @author Santiago Pericas-Geertsen
 *
 */

package org.apache.xalan.xsltc.dom;

import java.util.StringTokenizer;

import org.apache.xalan.xsltc.DOM;
import org.apache.xalan.xsltc.runtime.Hashtable;
import org.apache.xalan.xsltc.util.IntegerArray;
import org.apache.xml.dtm.DTM;
import org.apache.xml.dtm.DTMAxisIterator;
import org.apache.xml.dtm.ref.DTMAxisIteratorBase;

public class KeyIndex extends DTMAxisIteratorBase {

    /**
     * A mapping between values and nodesets.
     */
    private Hashtable _index = new Hashtable();

    /**
     * The node set associated to the current value passed
     * to lookupKey();
     */
    private IntegerArray _nodes = null;

    /**
     * The XSLTC DOM object if this KeyIndex is being used to implement the
     * id() function.
     */
    private DOM        _dom;
    
    private SAXImpl    _saxImpl;

    /**
     * Store position after call to setMark()
     */
    private int _markedPosition = 0;

    public KeyIndex(int dummy) {
    }

    public void setRestartable(boolean flag) {
    }

    /**
     * Adds a node to the node list for a given value. Nodes will
     * always be added in document order.
     */
    public void add(Object value, int node) {
	IntegerArray nodes;
	if ((nodes = (IntegerArray) _index.get(value)) == null) {
	    _index.put(value, nodes = new IntegerArray());
	}
	nodes.add(node);
    }

    /**
     * Merge the current value's nodeset set by lookupKey() with _nodes.
     */
    public void merge(KeyIndex other) {
	if (other == null) return;

	if (other._nodes != null) {
	    if (_nodes == null) {
		_nodes = other._nodes;
	    }
	    else {
		_nodes.merge(other._nodes);
	    }
	}
    }

    /**
     * This method must be called by the code generated by the id() function
     * prior to returning the node iterator. The lookup code for key() and
     * id() differ in the way the lookup value can be whitespace separated
     * list of tokens for the id() function, but a single string for the
     * key() function.
     */
    public void lookupId(Object value) {
	// Clear _nodes array
	_nodes = null;

	final StringTokenizer values = new StringTokenizer((String) value);
	while (values.hasMoreElements()) {
            final String token = (String) values.nextElement();
	    IntegerArray nodes = (IntegerArray) _index.get(token);

            if (nodes == null && _saxImpl != null && _saxImpl.hasDOMSource()) {
                nodes = getDOMNodeById(token);
            }

	    if (nodes == null) continue;

	    if (_nodes == null) {
		_nodes = nodes;
	    }
	    else {
		_nodes.merge(nodes);
	    }
	}
    }

    /**
     * Return an IntegerArray for the DOM Node which has the given id.
     * 
     * @param id The id
     * @return A IntegerArray representing the Node whose id is the given value.
     */
    public IntegerArray getDOMNodeById(String id) {
        IntegerArray nodes = null;
        if (_saxImpl != null) {
            int ident = _saxImpl.getElementById(id);
            if (ident != DTM.NULL) {
	        nodes = new IntegerArray();
	    	_index.put(id, nodes);
		nodes.add(ident);
            }
        }
        return nodes; 	
    }
    
    /**
     * This method must be called by the code generated by the key() function
     * prior to returning the node iterator.
     */
    public void lookupKey(Object value) {
	_nodes = (IntegerArray) _index.get(value);
	_position = 0;
    }

    /** 
     * Callers should not call next() after it returns END.
     */
    public int next() {
	if (_nodes == null) return DTMAxisIterator.END;

	return (_position < _nodes.cardinality()) ? 
	    _dom.getNodeHandle(_nodes.at(_position++)) : DTMAxisIterator.END;
    }

    public int containsID(int node, Object value) { 
	final String string = (String)value;
	if (string.indexOf(' ') > -1) {
	    final StringTokenizer values = new StringTokenizer(string);

	    while (values.hasMoreElements()) {
                final String token = (String) values.nextElement();
		IntegerArray nodes = (IntegerArray) _index.get(token);

		if (nodes == null && _saxImpl != null && _saxImpl.hasDOMSource()) {
		    nodes = getDOMNodeById(token);	
		}
		if (nodes != null && nodes.indexOf(node) >= 0) {
		    return 1;
		}
	    }
	    return 0;
	}
	else {
	    IntegerArray nodes = (IntegerArray) _index.get(value);
            if (nodes == null && _saxImpl != null && _saxImpl.hasDOMSource()) {
                nodes = getDOMNodeById(string);
            }
	    return (nodes != null && nodes.indexOf(node) >= 0) ? 1 : 0;
	}
    }

    public int containsKey(int node, Object value) { 
	final IntegerArray nodes = (IntegerArray) _index.get(value);
	return (nodes != null && nodes.indexOf(node) >= 0) ? 1 : 0;
    }

    /**
     * Resets the iterator to the last start node.
     */
    public DTMAxisIterator reset() {
	_position = 0;
	return this;
    }

    /**
     * Returns the number of elements in this iterator.
     */
    public int getLast() {
	return (_nodes == null) ? 0 : _nodes.cardinality();
    }

    /**
     * Returns the position of the current node in the set.
     */
    public int getPosition() {
	return _position;
    }

    /**
     * Remembers the current node for the next call to gotoMark().
     */
    public void setMark() {
	_markedPosition = _position;
    }

    /**
     * Restores the current node remembered by setMark().
     */
    public void gotoMark() {
	_position = _markedPosition;
    }

    /** 
     * Set start to END should 'close' the iterator, 
     * i.e. subsequent call to next() should return END.
     */
    public DTMAxisIterator setStartNode(int start) {
	if (start == DTMAxisIterator.END) {
	    _nodes = null;
	}
	else if (_nodes != null) {
	    _position = 0;
	}
	return (DTMAxisIterator) this;
    }
    
    /** 
     * Get start to END should 'close' the iterator, 
     * i.e. subsequent call to next() should return END.
     */
    public int getStartNode() {      
        return 0;
    }

    /**
     * True if this iterator has a reversed axis.
     */
    public boolean isReverse() {
	return(false);
    }

    /**
     * Returns a deep copy of this iterator.
     */
    public DTMAxisIterator cloneIterator() {
	KeyIndex other = new KeyIndex(0);
	other._index = _index;
	other._nodes = _nodes;
	other._position = _position;
	return (DTMAxisIterator) other;
    }
    
    public void setDom(DOM dom) {
    	_dom = dom;
    	if (dom instanceof SAXImpl) {
    	    _saxImpl = (SAXImpl)dom;
    	}
    	else if (dom instanceof DOMAdapter) {
    	    DOM idom = ((DOMAdapter)dom).getDOMImpl();
    	    if (idom instanceof SAXImpl) {
    	        _saxImpl = (SAXImpl)idom;
    	    }
    	}
    }
}
