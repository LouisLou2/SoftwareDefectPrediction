/*
 * Copyright 1999-2004 The Apache Software Foundation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * $Id: DOMEnhancedForDTM.java,v 1.2 2004/02/16 20:54:58 minchau Exp $
 */
package org.apache.xalan.xsltc;

/**
 * Interface for SAXImpl which adds methods used at run-time, over and above
 * those provided by the XSLTC DOM interface. An attempt to avoid the current
 * "Is the DTM a DOM, if so is it a SAXImpl, . . .
 * which was producing some ugly replicated code
 * and introducing bugs where that multipathing had not been
 * done.  This makes it easier to provide other DOM/DOMEnhancedForDTM
 * implementations, rather than hard-wiring XSLTC to SAXImpl.
 * 
 * @author Joseph Kesselman
 *
 */
public interface DOMEnhancedForDTM extends DOM {
    public short[] getMapping(String[] names, String[] uris, int[] types);
    public int[] getReverseMapping(String[] names, String[] uris, int[] types);
    public short[] getNamespaceMapping(String[] namespaces);
    public short[] getReverseNamespaceMapping(String[] namespaces);
    public String getDocumentURI();
    public void setDocumentURI(String uri);    
    public int getExpandedTypeID2(int nodeHandle);
    public boolean hasDOMSource();
    public int getElementById(String idString);    
}
