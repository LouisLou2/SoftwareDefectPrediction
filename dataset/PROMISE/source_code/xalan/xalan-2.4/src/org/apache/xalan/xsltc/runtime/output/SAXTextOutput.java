/*
 * @(#)$Id: SAXTextOutput.java,v 1.1 2002/06/10 15:45:59 santiagopg Exp $
 *
 * The Apache Software License, Version 1.1
 *
 *
 * Copyright (c) 2001 The Apache Software Foundation.  All rights
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
 *    distribution.
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
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES{} LOSS OF
 * USE, DATA, OR PROFITS{} OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
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
 * @author Santiago Pericas-Geertsen
 *
 */

package org.apache.xalan.xsltc.runtime.output;

import org.xml.sax.ContentHandler;
import org.xml.sax.ext.LexicalHandler;
import org.xml.sax.SAXException;

import org.apache.xalan.xsltc.TransletException;

public class SAXTextOutput extends SAXOutput {

    public SAXTextOutput(ContentHandler handler, String encoding) 
    {
    	super(handler, encoding);
    }

    public SAXTextOutput(ContentHandler handler, LexicalHandler lex, 
        String encoding)
    {
        super(handler, lex, encoding);
    }

    public void startDocument() throws TransletException { 
	try {
	    _saxHandler.startDocument();
	}
	catch (SAXException e) {
	    throw new TransletException(e);
	}
    }

    public void endDocument() throws TransletException { 
	try {
	    _saxHandler.endDocument();
	}
	catch (SAXException e) {
	    throw new TransletException(e);
	}
    }

    public void startElement(String elementName) 
	throws TransletException 
    {
    }

    public void endElement(String elementName) 
	throws TransletException 
    {
    }

    public void characters(String characters) 
	throws TransletException 
    { 
	try {
	    _saxHandler.characters(characters.toCharArray(), 0, 
		characters.length());
	}
	catch (SAXException e) {
	    throw new TransletException(e);
	}
    }

    public void characters(char[] characters, int offset, int length)
	throws TransletException 
    { 
	try {
	    _saxHandler.characters(characters, offset, length);
	}
	catch (SAXException e) {
	    throw new TransletException(e);
	}
    }

    public void comment(String comment) throws TransletException {
    }

    public void attribute(String name, String value) 
	throws TransletException 
    {
    }

    public void processingInstruction(String target, String data) 
	throws TransletException
    {
    }
}

